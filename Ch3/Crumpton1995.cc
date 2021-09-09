//
// Created by gbm on 28/06/19.
//

#include <fstream>
#include <iostream>
#include <cmath>

#include "DarcyBase.h"

using namespace dealii;
using namespace std;

/// Simulates the two different Hydraulic Conductivities
template <int dim>
Tensor<2, dim> conductivity( bool k1 ){
    Tensor<2, dim> matrix;
    if(k1)
        for (unsigned int d = 0; d < dim; ++d)
            matrix[d][d] = 1;
    else
        for (unsigned int d = 0; d < dim; ++d) {
            matrix[d][d] = 2;
            matrix[(dim - 1) - d][d] = 1;
        }
    return matrix;
}

template<int dim>
class SourceTerm : public Function<dim> {
public: SourceTerm() : Function<dim>(1) {}
    virtual void value_list(const vector<Point<dim>> &points, vector<double> & values,  bool k1) const {
        for (unsigned int p = 0; p < points.size(); ++p) {
            values[p] = 2*M_PI * M_PI;
            if( k1 ) {
                for (unsigned int d = 0; d < dim; ++d)
                    values[p] *= sin(M_PI * points[p][d]);
                values[p] *= 2;
            } else {
                double aux1 = 1.0;
                double aux2 = 1.0;
                for (unsigned int d = 0; d < dim; ++d) {
                    aux1 *= sin(M_PI * points[p][d]);
                    aux2 *= cos(M_PI * points[p][d]);
                }
                values[p] *= (2*aux1 - aux2);
            }
        }
    }
};

template <int dim>
class DirichletBoundary : public Function<dim> {
public: DirichletBoundary() : Function<dim>(1) {  }
    virtual double value(const Point<dim> &p, const unsigned int  /*component = 0*/ ) const {
        double return_value = 2.0;
        for (int i = 0; i < dim; ++i)
            return_value *= sin(M_PI * p[i]);

        return return_value;
    }
};

template <int dim>
class ExactSolutionK1 : public Function<dim>{
public:
    ExactSolutionK1 () : Function<dim>(dim+1) {}
    virtual void vector_value (const Point<dim> &p, Vector<double>  &values) const  {

        values[dim] = 2.0;
        for (int i = 0; i < dim; ++i) {
            values[dim] *= sin(M_PI * p[i]);
        }
        values[0] = -2*M_PI*cos(M_PI * p[0])*sin(M_PI * p[1]);
        values[1] = -2*M_PI*sin(M_PI * p[0])*cos(M_PI * p[1]);
    }
};

template <int dim>
class ExactSolutionK2 : public Function<dim>{
public:
    ExactSolutionK2 () : Function<dim>(dim+1) {}
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const  {

        values[dim] = 1.0;
        for (int i = 0; i < dim; ++i) {
            values[dim] *= sin(M_PI * p[i]);
        }
        values[0] = -M_PI * ( 2*cos(M_PI * p[0])*sin(M_PI * p[1]) +   sin(M_PI * p[0])*cos(M_PI * p[1]) );
        values[1] = -M_PI * (   cos(M_PI * p[0])*sin(M_PI * p[1]) + 2*sin(M_PI * p[0])*cos(M_PI * p[1]) );
    }
};

template <int dim>
class Crumpton : DarcyBase<dim> {
public:

    Crumpton(const unsigned int degree) : DarcyBase<dim>(degree) {}

    void run(int refinement, ConvergenceTable &convergence_table) {
        make_grid_and_dofs(refinement, 0.0, 1.0 );
        assemble_system(true);
        this->solveBiCGstab(1e-10);
//        this->solve();
        assemble_system(false);
//        this->compute_errors(convergence_table ,  ExactSolution<dim>() );
//        compute_errors_by_domain( convergence_table );
//        this->output_results(refinement);
    }

private:

    const SourceTerm<dim> source_term;
    string filename_dir = "output/Crumpton/";

    /// Check which domain the cell is at
    bool isK1( const typename DoFHandler<dim>::cell_iterator &c){
        return (c->center()[0] <= -1.0 || c->center()[0] >=  1.0 || c->center()[1] <= -1.0 || c->center()[1] >=  1.0);
    }

    void make_grid_and_dofs(int numberCells, double inf_domain, double sup_domain) {

        GridGenerator::hyper_cube(this->triangulation, inf_domain, sup_domain);
        this->triangulation.refine_global(numberCells);

        this->dof_handler.distribute_dofs(this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        cout << "Number of active cells: \t\t\t" << this->triangulation.n_active_cells() << endl
             << "Number of degrees of freedom for the multiplier: " << this->dof_handler.n_dofs() << endl << endl;

        this->solution.reinit(this->dof_handler.n_dofs());
        this->system_rhs.reinit(this->dof_handler.n_dofs());
        this->solution_local.reinit(this->dof_handler_local.n_dofs());

        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern(this->dof_handler, dsp, this->constraints, false);
        this->sparsity_pattern.copy_from( dsp );
        this->system_matrix.reinit(this->sparsity_pattern);

        this->constraints.clear();
        DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
        map<types::boundary_id, const Function<dim> *> boundary_functions;
        VectorTools::interpolate_boundary_values(this->dof_handler, 0 , DirichletBoundary<dim>(), this->constraints );
        this->constraints.close();
    }

    void assemble_system( bool globalProblem ) {

        const double delta  =  0.5;
        const double delta1 = -delta;
        const double delta2 =  delta;
        const double delta3 =  delta;

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

        vector<double>            st_values( n_q_points      );
        vector<double>          mult_values( n_face_q_points );

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

            bool is_k1 = isK1(cell);
            const Tensor<2,dim> k_matrix = conductivity<dim>( is_k1 );
            auto k_inverse = invert( k_matrix );

            source_term.value_list(fe_values_local.get_quadrature_points(), st_values, is_k1);


            for (unsigned int q = 0; q < n_q_points; ++q) {
                const double JxW = fe_values_local.JxW(q);
                for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                        auto curl2d_i = k_inverse[1][0]*fe_values_local.shape_grad_component(i, q, 0)[0] -
                                        k_inverse[0][1]*fe_values_local.shape_grad_component(i, q, 1)[1] +
                                        k_inverse[1][1]*fe_values_local.shape_grad_component(i, q, 1)[0] -
                                        k_inverse[0][0]*fe_values_local.shape_grad_component(i, q, 0)[1];

                    const Tensor<1, dim> phi_i_u = fe_values_local[velocities].value(i, q);
                    const double div_phi_i_u = fe_values_local[velocities].divergence(i, q);
//                    auto curl_phi_i_u = fe_values_local[velocities].curl(i, q);
                    const double phi_i_p = fe_values_local[pressure].value(i, q);
                    const Tensor<1, dim> grad_phi_i_p = fe_values_local[pressure].gradient(i, q);
                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                            auto curl2d_j = k_inverse[1][0]*fe_values_local.shape_grad_component(j, q, 0)[0] -
                                            k_inverse[0][1]*fe_values_local.shape_grad_component(j, q, 1)[1] +
                                            k_inverse[1][1]*fe_values_local.shape_grad_component(j, q, 1)[0] -
                                            k_inverse[0][0]*fe_values_local.shape_grad_component(j, q, 0)[1];

                        const Tensor<1, dim> phi_j_u = fe_values_local[velocities].value(j, q);
                        const double div_phi_j_u = fe_values_local[velocities].divergence(j, q);
//                        auto curl_phi_j_u = fe_values_local[velocities].curl(j, q);
                        const double phi_j_p = fe_values_local[pressure].value(j, q);
                        const Tensor<1, dim> grad_phi_j_p = fe_values_local[pressure].gradient(j, q);

                        A_matrix(i, j) +=           (   k_inverse * phi_i_u*phi_j_u               // A * <u,v>
                                                    -   div_phi_i_u*phi_j_p                       // <p, div v>
                                                    -   phi_i_p * div_phi_j_u                     // <div u, q>
                                                    ) * JxW;

                        A_matrix(i, j) += delta1 *  (   k_inverse * phi_i_u * phi_j_u                 // d1 * A * <u,v>
                                                    +   phi_i_u      * grad_phi_j_p                   // d1 * <grad(p), v>
                                                    +   grad_phi_i_p * phi_j_u                        // d1 * <u, grad(q)>
                                                    +   k_matrix * grad_phi_i_p * grad_phi_j_p        // d1 * K <grad(p), grad(q)>
                                                    ) * JxW;

                        A_matrix(i, j) += delta2 * ( linfty_norm(k_inverse) * div_phi_i_u * div_phi_j_u ) * JxW;  // d2 * |A| * <div u, div v>

//                        A_matrix(i, j) += delta3 * ( linfty_norm(k_inverse) *curl_phi_i_u *curl_phi_j_u ) * JxW;  // d3 * |A| * <rot u, rot v>
                        A_matrix(i, j) += delta3 * ( linfty_norm( k_matrix ) * curl2d_i * curl2d_j ) * JxW;  // d3 * |K| * <A rot u, A rot v>

                    }

                        F_vector(i) += (-phi_i_p + delta2 * linfty_norm(k_inverse) * div_phi_i_u) * st_values[q] * JxW;        // <f, (-q + d2 * |A| * div v)>
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
    void compute_errors_by_domain (ConvergenceTable &convergence_table) {
        const ComponentSelectFunction<dim> p_mask (dim, dim + 1);
        const ComponentSelectFunction<dim> u_mask(make_pair(0, dim), dim + 1);

        const uint n_active_cells = this->triangulation.n_active_cells();

        ExactSolutionK1<dim> exact_solution_K1;
        ExactSolutionK2<dim> exact_solution_K2;
        Vector<double> p_errors_K1   (n_active_cells);
        Vector<double> p_errors_K2   (n_active_cells);
        Vector<double> p_errors      (n_active_cells);
        Vector<double> u_errors_K1   (n_active_cells);
        Vector<double> u_errors_K2   (n_active_cells);
        Vector<double> u_errors      (n_active_cells);

        double u_l2_error, p_l2_error;

        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, exact_solution_K1,
                                           p_errors_K1, this->quadrature, VectorTools::L2_norm, &p_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, exact_solution_K1,
                                           u_errors_K1, this->quadrature, VectorTools::L2_norm, &u_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, exact_solution_K2,
                                           p_errors_K2, this->quadrature, VectorTools::L2_norm, &p_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, exact_solution_K2,
                                           u_errors_K2, this->quadrature, VectorTools::L2_norm, &u_mask);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            unsigned int i = cell->index();
            if( isK1 (cell) ) {
                p_errors_K2(i) = 0.;
                u_errors_K2(i) = 0.;
            }else {
                p_errors_K1(i) = 0.;
                u_errors_K1(i) = 0.;
            }
        }
        for (int i=0; i < n_active_cells; i++) {
            p_errors(i) = p_errors_K1(i) + p_errors_K2(i);
            u_errors(i) = u_errors_K1(i) + u_errors_K2(i);
        }

        p_l2_error = VectorTools::compute_global_error(this->triangulation, p_errors, VectorTools::L2_norm);
        u_l2_error = VectorTools::compute_global_error(this->triangulation, u_errors, VectorTools::L2_norm);

        convergence_table.add_value("cells", n_active_cells );
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);
    }
};

int main() {
    const int dim = 2;
    for (int degree = 1; degree <= 3; ++degree) {
        ConvergenceTable convergence_table;

        for (int i = 2; i <= 7; ++i) {
            cout << "   ---   Refinement #" << i << "   ---   " << endl;
            Crumpton<dim> mixed_laplace_problem(degree);
            mixed_laplace_problem.run(i, convergence_table);
        }
        show_convergence(convergence_table, dim, degree, "Crump_DG_D");
    }
    return 0;
}
