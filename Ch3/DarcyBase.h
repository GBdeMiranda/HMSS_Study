//
// Created by gbm on 16/08/2020.
//

#ifndef HYBRIDMIXED_DARCYBASE_H
#define HYBRIDMIXED_DARCYBASE_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_face.h>   // Para problema com multiplicador descontinuo
#include <deal.II/fe/fe_trace.h>  // Para problema com multiplicador continuo

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>

using namespace dealii;
using namespace std;

template <int dim>
class DarcyBase {
public:
    DarcyBase(const unsigned int degree) : degree(degree),
//                fe_local( FE_DGQLegendre<dim>(degree),   dim, FE_DGQLegendre<dim>(degree), 1 ),     // Stabilized
                fe_local( FE_DGRaviartThomas<dim>(degree), 1, FE_DGQLegendre<dim>(degree), 1 ),     // Stable
                dof_handler_local(triangulation), quadrature(degree + 2),
                fe(degree ), dof_handler(triangulation), face_quadrature(degree + 2)
//                fe(degree-1), dof_handler(triangulation), face_quadrature( degree )
                { }

protected :

    const unsigned int  degree;
    Triangulation<dim>  triangulation;
    FESystem<dim>       fe_local;
    DoFHandler<dim>     dof_handler_local;
    Vector<double>      solution_local;

    FE_FaceQ<dim>   fe;                                               // Discontinuous Multiplier
//    FE_TraceQ<dim>  fe;                                              // Continuous Multiplier

    QGauss<dim  >      quadrature;
    QGauss<dim-1> face_quadrature;

    DoFHandler<dim> dof_handler;
    Vector<double>  solution;
    Vector<double>  system_rhs;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    ConstraintMatrix        constraints;

    void static_condensation( typename DoFHandler<dim>::active_cell_iterator cell,
                          FullMatrix<double> A, FullMatrix<double> B, FullMatrix<double> BT,
                          FullMatrix<double> C, Vector<double> F, Vector<double> G) {

        A.gauss_jordan();                                   //   A = A^{-1}
        FullMatrix<double>              aux_matrix(fe_local.dofs_per_cell, fe.dofs_per_cell );
        Vector<double>                  aux_vector(fe_local.dofs_per_cell                   );
        vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

        BT.copy_transposed(B);

        A.vmult(aux_vector, F, false);            //       A^{-1} * F
        BT.vmult(G, aux_vector, true);            // B.T * A^{-1} * F
        A.mmult(aux_matrix, B, false);            //       A^{-1} * B
        BT.mmult(C, aux_matrix, true);            // B.T * A^{-1} * B - C

        cell->get_dof_indices(dof_indices);
        this->constraints.distribute_local_to_global(C, G, dof_indices, this->system_matrix, this->system_rhs);
    }

    void solveBiCGstab( const double tol ) {
        Timer timer;

        SolverControl solver_control(system_matrix.m() * 10, tol);
        SolverBicgstab<> solver( solver_control );
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity() );

        timer.stop();
//        cout  << "Solved (" << timer.cpu_time() << "s)" << endl;

        cout << "Number of iterations: " << solver_control.last_step() << endl << endl;

        constraints.distribute(solution);
    }

    void solveGMRES( double tol ) {
        Timer timer;

        SolverControl solver_control(system_matrix.m() * 10, tol);
        SolverGMRES<> solver( solver_control );
        solver.solve(system_matrix, solution, system_rhs, PreconditionIdentity() );

        timer.stop();
//        cout  << "Solved (" << timer.cpu_time() << "s)" << endl;

        cout << "Number of iterations: " << solver_control.last_step() << endl << endl;

        constraints.distribute(solution);
    }

    void solve( ) {
        Timer timer;
        
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);

        timer.stop();
        cout  << "Solved (" << timer.cpu_time() << "s)" << endl << endl;
        
        constraints.distribute(solution);
    }

    void output_results( const unsigned int refinements, string filename_dir = "" ) const {
        string filename = "output/Darcy/" + filename_dir + "solution_local_" + Utilities::to_string(refinements, 0) + "_d"
                          + dealii::Utilities::to_string(degree, 0) + ".vtu" ;
        ofstream output(filename.c_str());

        DataOut<dim> data_out;
        vector<string> names(dim, "Velocity");
        names.emplace_back("Pressure");
        vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation
                (dim + 1, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
        data_out.add_data_vector(dof_handler_local, solution_local, names, component_interpretation);

        data_out.build_patches(fe_local.degree);
        data_out.write_vtu(output);

        string filename_face = filename_dir + Utilities::to_string(refinements, 0) + "_d"
                               + Utilities::to_string(degree, 0) + " .vtu";
        ofstream face_output(filename_face.c_str());

        DataOutFaces<dim> data_out_face(false);
        vector<string> face_name(1, "Trace_pressure");
        vector<DataComponentInterpretation::DataComponentInterpretation>
                face_component_type(1, DataComponentInterpretation::component_is_scalar);

        data_out_face.add_data_vector(dof_handler, solution, face_name, face_component_type);

        data_out_face.write_vtu(face_output);
    }

    void compute_errors(ConvergenceTable &convergence_table, const Function<dim> &exact_solution) {
        const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> velocity_mask( make_pair(0, dim), dim + 1);

        Vector<double> cellwise_errors(triangulation.n_active_cells());

        QGauss<dim> quadrature(degree + 2);

        double u_l2_error, p_l2_error;

        VectorTools::integrate_difference(dof_handler_local, solution_local, exact_solution, cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

        VectorTools::integrate_difference(dof_handler_local, solution_local, exact_solution, cellwise_errors, quadrature, VectorTools::L2_norm, &pressure_mask);
        p_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

        convergence_table.add_value("cells", triangulation.n_active_cells());
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);
    }
};

void show_convergence( ConvergenceTable &convergence_table , int dim, string filename_dir = "" ) {
    convergence_table.set_precision("L2_u", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.evaluate_convergence_rates("L2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_p", 3);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.evaluate_convergence_rates("L2_p", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.write_text(cout);

    string filenametex = filename_dir + "ConvergenceRates";
    ofstream  tex_output(filenametex + ".tex");
    convergence_table.write_tex(tex_output);
    ofstream data_output(filenametex + ".dat");
    convergence_table.write_text(data_output);
}

#endif //HYBRIDMIXED_DARCYBASE_H