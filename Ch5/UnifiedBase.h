#ifndef TRANSPORTBASE_H
#define TRANSPORTBASE_H

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/parameter_handler.h>
#include <deal.II/base/parsed_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_bicgstab.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/affine_constraints.h>

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

#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii; using namespace std;
template <int dim>
class UnifiedBase {
public:
    UnifiedBase(const unsigned int degree) : degree(degree), dof_handler_local(triangulation), quadrature(degree + 2),
//                fe_local(FE_DGQ<dim>(degree), dim, FE_DGQ<dim>(degree), 1 ),
                fe_local(FE_DGRaviartThomas<dim>(degree), 1, FE_DGQ<dim>(degree), 1 ),
                fe(degree), dof_handler(triangulation), face_quadrature(degree + 2)
                , dof_handler_perm(triangulation), fe_perm(degree)  /// Permeability Field
                {}
protected :
    const unsigned int  degree;
    Triangulation<dim>  triangulation;

    QGauss<dim  >      quadrature;
    QGauss<dim-1> face_quadrature;

    FE_DGQ<dim> fe_perm;                  /// Permeability Field
    DoFHandler<dim> dof_handler_perm;     /// Permeability Fieldq
    Vector<double> permeability_local;    /// Permeability Field
    ConstraintMatrix constraints2;        /// Permeability Field

//    FE_TraceQ<dim>  fe;                                                 // Continuous Multiplier
    FE_FaceQ<dim>   fe;                                                 // Discontinuous Multiplier

    DoFHandler<dim> dof_handler;

    Vector<double>  solution;
    Vector<double>  system_rhs;

    FESystem<dim>       fe_local;
    DoFHandler<dim>     dof_handler_local;

    Vector<double>      transp_solution_local;
    Vector<double>      darcy_solution_local;

    Vector<double>      transp_solution_local_mid;

    SparsityPattern         sparsity_pattern;
    SparseMatrix<double>    system_matrix;
    ConstraintMatrix        constraints;
//    AffineConstraints<double> constraints;

    void static_condensation( typename DoFHandler<dim>::active_cell_iterator cell,
                              FullMatrix<double> A, FullMatrix<double> B, FullMatrix<double> BT,
                              FullMatrix<double> C, Vector<double> F, Vector<double> G) {

        A.gauss_jordan();                                   //   A = A^{-1}
        FullMatrix<double>              aux_matrix(fe_local.dofs_per_cell, fe.dofs_per_cell );
        Vector<double>                  aux_vector(fe_local.dofs_per_cell                   );
        vector<types::global_dof_index> dof_indices(fe.dofs_per_cell);

        A.vmult(aux_vector, F, false);            //       A^{-1} * F
        BT.vmult(G, aux_vector, true);            // B.T * A^{-1} * F
        A.mmult(aux_matrix, B, false);            //       A^{-1} * B
        BT.mmult(C, aux_matrix, true);            // B.T * A^{-1} * B - C

        cell->get_dof_indices(dof_indices);
        this->constraints.distribute_local_to_global(C, G, dof_indices, this->system_matrix, this->system_rhs);
    }

    void solveGMRES( double tol ) {
//        Timer timer;

        PreconditionJacobi<SparseMatrix<double> > precondition;
        precondition.initialize(system_matrix, .6);

        SolverControl solver_control(system_matrix.m() * 10, tol);
        SolverGMRES<> solver( solver_control );
        solver.solve(system_matrix, solution, system_rhs, precondition );

//        timer.stop();
//        cout  << "Solved (" << timer.cpu_time() << "s)" << endl;
        cout << "Number of iterations: " << solver_control.last_step() << endl << endl;
    }

    void solve( ) {
//        Timer timer;

        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);

//        timer.stop();
//        cout  << "Solved in " << timer.cpu_time() << "s" << endl;

        constraints.distribute(solution);
    }

    void output_results (bool Transport, const string& dirname = "" , const string& filename_extra = "" ) const {

        vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation
                (dim+1, DataComponentInterpretation::component_is_part_of_vector);
        component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;

        DataOut<dim> data_out;
        string filename;
        if(Transport)   {
            filename = dirname + "Transport" + filename_extra +  ".vtu";
            vector<string> names (dim, "vector");
            names.emplace_back("scalar");

            data_out.add_data_vector (dof_handler_local, transp_solution_local, names, component_interpretation);
        } else  {
            filename = dirname + "Darcy" + filename_extra + ".vtu";
            vector<string> names (dim, "vector");
            names.emplace_back("scalar");

            data_out.add_data_vector (dof_handler_local, darcy_solution_local, names, component_interpretation);
        }

        ofstream output (filename.c_str());
        data_out.build_patches (fe_local.degree);
        data_out.write_vtu (output);
    }

    void output_perms( const string& dirname = "" , const string& filename_extra = "") const { /// Permeability Field
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler_perm);
        data_out.add_data_vector(permeability_local, "perm");

        data_out.build_patches();
        string filename = dirname + "Permeability" + filename_extra + ".vtu";
        ofstream output(filename.c_str());
        data_out.build_patches (fe_perm.degree);
        data_out.write_vtu(output);
    }

    void compute_errors(ConvergenceTable &convergence_table,
                        const Function<dim> &exact_solution_Darcy, const Function<dim> &exact_solution_Transport) {
                        const ComponentSelectFunction<dim> pressure_mask(dim, dim + 1);
                        const ComponentSelectFunction<dim> velocity_mask(make_pair(0, dim), dim + 1);


        Vector<double> cellwise_errors(triangulation.n_active_cells());

        QGauss<dim> quadrature(degree + 2);

        double u_l2_error, p_l2_error, c_l2_error;

        VectorTools::integrate_difference(dof_handler_local, darcy_solution_local, exact_solution_Darcy, cellwise_errors, quadrature, VectorTools::L2_norm, &velocity_mask);
        u_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

        VectorTools::integrate_difference(dof_handler_local, darcy_solution_local, exact_solution_Darcy, cellwise_errors, quadrature, VectorTools::L2_norm, &pressure_mask);
        p_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

        VectorTools::integrate_difference(dof_handler_local, transp_solution_local, exact_solution_Transport, cellwise_errors, quadrature, VectorTools::L2_norm, &pressure_mask);
        c_l2_error = VectorTools::compute_global_error(triangulation, cellwise_errors, VectorTools::L2_norm);

        convergence_table.add_value("cells", triangulation.n_active_cells());
//        convergence_table.add_value("delta", d1); /// Studying the delta choice
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);
        convergence_table.add_value("L2_c", c_l2_error);
    }
};

void show_convergence( ConvergenceTable &convergence_table , int dim, const string& filename_extra = "" ) {
    convergence_table.set_precision("L2_u", 3);
    convergence_table.set_scientific("L2_u", true);
    convergence_table.evaluate_convergence_rates("L2_u", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_p", 3);
    convergence_table.set_scientific("L2_p", true);
    convergence_table.evaluate_convergence_rates("L2_p", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.set_precision("L2_c", 3);
    convergence_table.set_scientific("L2_c", true);
    convergence_table.evaluate_convergence_rates("L2_c", "cells", ConvergenceTable::reduction_rate_log2, dim);

    convergence_table.write_text( cout);

    string filenametex = "output/ConvRates/taxas_" + filename_extra;
//    ofstream tex_output(filenametex + ".tex");
//    convergence_table.write_tex(tex_output);
    ofstream data_output(filenametex + ".dat");
    convergence_table.write_text(data_output, TableHandler::org_mode_table );
}

#endif //TRANSPORTBASE_H