//
// Created by gbm on 28/06/19.
//

#include <fstream>
#include <iostream>
#include <cmath>

#include "DarcyBase.h"

using namespace dealii;
using namespace std;

template<int dim>
class SourceTerm : public Function<dim> {
public: SourceTerm() : Function<dim>(1) {}
    virtual double value( const Point<dim> &p , const unsigned int  /*component = 0*/) const {
        double return_value = dim * M_PI * M_PI;
        for (int i = 0; i < dim; ++i)       return_value *= sin(M_PI * p[i]);
        return return_value;
    }
};

template<int dim>
class DirichletBoundary : public Function<dim> {
public: DirichletBoundary() : Function<dim>(1) {}
    virtual double value( const Point<dim> &p , const unsigned int  /*component = 0*/) const {
        double return_value = 1.0;
        for (int i = 0; i < dim; ++i)       return_value *= sin(M_PI * p[i]);
        return return_value;
    }
};

template<int dim>
class ExactSolution : public Function<dim> {
public: ExactSolution() : Function<dim>(dim + 1) {}
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {

        for (unsigned int i = 0; i < dim; ++i) {
            values[i] = -M_PI;
            for (unsigned int j = 0; j < dim; ++j) {
                if (i == j)                 values[i] *= cos(M_PI * p[j]);
                else                        values[i] *= sin(M_PI * p[j]);
            }
        }
        values[dim] = 1.0;
        for (int i = 0; i < dim; ++i)       values[dim] *= sin(M_PI * p[i]);
    }

};

template <int dim>
class DarcyHomogeneous : DarcyBase<dim> {
public:

    DarcyHomogeneous(const unsigned int degree) : DarcyBase<dim>(degree) {}

    void run(int refinement, ConvergenceTable &convergence_table) {
        make_grid_and_dofs(refinement, 0.0, 1.0 );
        assemble_system(true);
//        this->solve();
        this->solveBiCGstab( 1e-10 );
        assemble_system(false);
        this->compute_errors(convergence_table, ExactSolution<dim>() );
//        this->output_results(refinement);
    }

private:

    SourceTerm<dim> source_term;
    string filename_dir = "output/Classic/";

    void make_grid_and_dofs(int numberCells, double inf_domain, double sup_domain) {
        GridGenerator::hyper_cube(this->triangulation, inf_domain, sup_domain);
        this->triangulation.refine_global(numberCells);

        this->dof_handler.distribute_dofs(this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        cout << "#Active Cells:  \t" << this->triangulation.n_active_cells() << endl
             << "Multiplier DoFs: \t" << this->dof_handler.n_dofs() << endl
             << "Local DoFs:      \t" << this->dof_handler_local.n_dofs() << endl << endl;

        this->solution.reinit(this->dof_handler.n_dofs());
        this->system_rhs.reinit(this->dof_handler.n_dofs());
        this->solution_local.reinit(this->dof_handler_local.n_dofs());

        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(this->dof_handler, dsp, this->constraints, false);
        this->sparsity_pattern.copy_from( dsp );
        this->system_matrix.reinit(this->sparsity_pattern);

        this->constraints.clear();
        DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
        std::map<types::boundary_id, const Function<dim> *> boundary_functions;
        VectorTools::interpolate_boundary_values(this->dof_handler, 0 , DirichletBoundary<dim>(), this->constraints );
        this->constraints.close();
    }

    void assemble_system( bool globalProblem ) {

        const double delta1 = -0.;
        const double delta2 =  0.;
        const double delta3 =  0.;

        FEValues<dim> fe_values_local(this->fe_local, this->quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(this->fe, this->face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values_local(this->fe_local, this->face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

        const unsigned int dofs_local_per_cell  = this->fe_local.dofs_per_cell;
        const unsigned int dofs_multiplier      = this->fe.dofs_per_cell;
        const unsigned int n_q_points           = this->quadrature.size();
        const unsigned int n_face_q_points      = this->face_quadrature.size();

        FullMatrix<double>  A_matrix    (dofs_local_per_cell, dofs_local_per_cell);
        FullMatrix<double>  B_matrix    (dofs_local_per_cell, dofs_multiplier);
        FullMatrix<double>  BT_matrix   (dofs_multiplier, dofs_local_per_cell);
        FullMatrix<double>  C_matrix    (dofs_multiplier, dofs_multiplier);
        Vector<double>      F_vector    (dofs_local_per_cell);
        Vector<double>      U_vector    (dofs_local_per_cell);
        Vector<double>      G_vector    (dofs_multiplier);

        vector<double>        st_values (n_q_points     );
        vector<double>      mult_values (n_face_q_points);

        const FEValuesExtractors::Vector velocities(0);
        const FEValuesExtractors::Scalar pressure(dim);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            typename DoFHandler<dim>::active_cell_iterator loc_cell(&this->triangulation, cell->level(), cell->index(), &this->dof_handler_local);

            fe_values_local.reinit(loc_cell);

            A_matrix    = 0.;
            F_vector    = 0.;
            B_matrix    = 0.;
            C_matrix    = 0.;
            BT_matrix   = 0.;
            U_vector    = 0.;
            G_vector    = 0.;

            source_term.value_list(fe_values_local.get_quadrature_points(), st_values);

            for (unsigned int q = 0; q < n_q_points; ++q) {
                const double JxW = fe_values_local.JxW(q);

                for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                    const Tensor<1, dim> phi_i_u = fe_values_local[velocities].value(i, q);
                    const double div_phi_i_u = fe_values_local[velocities].divergence(i, q);
                    auto curl_phi_i_u = fe_values_local[velocities].curl(i, q);
                    const double phi_i_p = fe_values_local[pressure].value(i, q);
                    const Tensor<1, dim> grad_phi_i_p = fe_values_local[pressure].gradient(i, q);

                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                        const Tensor<1, dim> phi_j_u = fe_values_local[velocities].value(j, q);
                        const double div_phi_j_u = fe_values_local[velocities].divergence(j, q);
                        auto curl_phi_j_u = fe_values_local[velocities].curl(j, q);
                        const double phi_j_p = fe_values_local[pressure].value(j, q);
                        const Tensor<1, dim> grad_phi_j_p = fe_values_local[pressure].gradient(j, q);

                        A_matrix(i, j) +=           (           phi_i_u *     phi_j_u     // A * <u,v>
                                                        -   div_phi_i_u *     phi_j_p     // <p, div v>
                                                        -       phi_i_p * div_phi_j_u     // <div u, q>
                                                    ) * JxW;

                        A_matrix(i, j) += delta1 *  (            phi_i_u *      phi_j_u   // d1 * A * <u,v>
                                                        +        phi_i_u * grad_phi_j_p   // d1 * <grad(p), v>
                                                        +   grad_phi_i_p *      phi_j_u   // d1 * <u, grad(q)>
                                                        +   grad_phi_i_p * grad_phi_j_p   // d1 * K <grad(p), grad(q)>
                                                    ) * JxW;

                        A_matrix(i, j) += delta2 * ( div_phi_i_u * div_phi_j_u ) * JxW;   // d2 * |A| * <div u, div v>

                        A_matrix(i, j) += delta3 * ( curl_phi_i_u *curl_phi_j_u ) * JxW;  // d3 * |A| * <rot u, rot v>

                    }
                    F_vector(i) += (-phi_i_p + delta2 * div_phi_i_u) * st_values[q] * JxW; // <f, (-q + d2 * |A| * div v)>
                }
            }
            if (globalProblem) {
                for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n) {

                    fe_face_values.reinit(cell, face_n);
                    fe_face_values_local.reinit(loc_cell, face_n);
                    for (unsigned int q = 0; q < n_face_q_points; ++q) {
                        const double JxW = fe_face_values.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values.normal_vector(q);
                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_u = fe_face_values_local[velocities].value(i, q);
                            for (unsigned int j = 0; j < dofs_multiplier; ++j) {
                                const double phi_j_m = fe_face_values.shape_value(j, q);
                                B_matrix(i, j) += phi_j_m * (phi_i_u * normal) * JxW;
                            }
                        }
                    }
                }
                this->static_condensation(cell,A_matrix,B_matrix,BT_matrix,C_matrix,F_vector,G_vector);
            } else {
                for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit(cell, face_n);
                    fe_face_values_local.reinit(loc_cell, face_n);

                    fe_face_values.get_function_values(this->solution, mult_values);

                    for (unsigned int q = 0; q < n_face_q_points; ++q) {
                        const double JxW = fe_face_values.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_u = fe_face_values_local[velocities].value(i, q);

                            F_vector(i) -= (phi_i_u * normal) * mult_values[q] * JxW;
                        }
                    }
                }
                A_matrix.gauss_jordan();     //               cout << " invert "  << endl;
                A_matrix.vmult(U_vector, F_vector, false);
                loc_cell->set_dof_values(U_vector, this->solution_local);
            }
        }
    }
};

int main () {
    const int dim = 3;
    const int degree = 1;
    ConvergenceTable convergence_table;

    for (int i = 2; i <= 5; ++i) {
        cout << "   ---   Refinement #" << i << "   ---   " << endl;
        DarcyHomogeneous<dim> problem( degree );
        problem.run( i, convergence_table );
    }
    show_convergence(convergence_table , dim );
    return 0;
}
