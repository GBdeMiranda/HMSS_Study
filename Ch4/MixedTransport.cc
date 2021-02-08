/*
 * Author: Gabriel Brandao de Miranda
 * Date: 28/06/19.
 */

#include <iostream>
#include <cmath>
#include <deal.II/dofs/dof_tools.h>
#include "MixedTransportBase.h"

double epsilon          = 1.0;
double porosity         = 1.0;
const double coef_conv  = 1.0;

template <int dim>
class SourceTerm : public Function<dim> {
public: SourceTerm() : Function<dim>() { }
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return coef_conv * dim * sin(M_PI_4) * exp( -this->get_time() ) * cos( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );
    }
};

template <int dim>
class InitialCondition : public Function<dim> {
public: InitialCondition() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        for(int i = 0; i < dim; i++) values[i] = 0.0;
        values[dim] = sin( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );
    }
};

template <int dim>
class ExactSolution : public Function<dim> {
public: ExactSolution() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        Assert (values.size() == dim+1, ExcDimensionMismatch (values.size(), dim+1));

        values[0] = -epsilon*sin(M_PI_4) * exp( -this->get_time() ) * cos( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );
        values[1] = -epsilon*sin(M_PI_4) * exp( -this->get_time() ) * cos( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );
        values[dim] = exp( -this->get_time() ) * sin( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );

    }
};

template <int dim>
class Boundary : public Function<dim> {
public: Boundary() : Function<dim>() { }
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return exp( -this->get_time() ) * sin( p[0] * sin(M_PI_4) + p[1] * sin(M_PI_4) );
    }
};

template <int dim>
class MixedTransport : MixedTransportBase<dim>{
public:
    MixedTransport(const unsigned int degree) : MixedTransportBase<dim>(degree) {}

    void run(const int refinements, ConvergenceTable &convergence_table, const double final_time ) {
        double current_time = 0.0;

        make_grid_and_dofs( refinements , 0.0 , 1.0);

        InitialCondition<dim> initial_condition;

        this->constraints.close();
        VectorTools::project( this->dof_handler_local,  this->constraints, QGauss<dim>( this->degree+3), initial_condition,  this->solution_local);

        while( current_time < final_time ) {
            cout << "   -> Current time:  " << current_time << endl << endl;
            current_time += dt;

            change_boundary(current_time);

            assemble_system( true, current_time);
            this->solve();
            assemble_system( false, current_time );

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);

            /// Only if using dt = h^{(k+1)/2}/2
            for(int i = 0; i < this->dof_handler_local.n_dofs(); i++)
                this->solution_local[i] = 2 * this->solution_local_mid[i] - this->solution_local[i];
            current_time += dt;   /// using dt = h^{(k+1)/2}/2
        }
        cout << "   -> Current time:  " << current_time << endl << endl;
        compute_errors_finaltime(convergence_table, current_time);
    }

private:
    double h_mesh;
    double dt;

    SourceTerm<dim>     source_term;

    void make_grid_and_dofs( int numberCells, double inf_domain, double sup_domain ) {
        GridGenerator::hyper_cube(this->triangulation, inf_domain, sup_domain);
        this->triangulation.refine_global(numberCells);

        this->dof_handler.distribute_dofs(this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        h_mesh = GridTools::maximal_cell_diameter(this->triangulation)*sqrt(2)/2;

//        dt = pow( h_mesh, this->degree + 1 );
        dt = pow( h_mesh, (this->degree + 1.0)/2 )/2; /// using dt = h^{(k+1)/2}

        this->solution.reinit(this->dof_handler.n_dofs());
        this->system_rhs.reinit(this->dof_handler.n_dofs());
        this->solution_local.reinit(this->dof_handler_local.n_dofs());
        this->solution_local_mid.reinit(this->dof_handler_local.n_dofs());

        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern( this->dof_handler, dsp, this->constraints, false );
        this->sparsity_pattern.copy_from( dsp );
        this->system_matrix.reinit(this->sparsity_pattern);

        cout << "Number of active cells: \t\t\t" << this->triangulation.n_active_cells() << endl
             << "Number of degrees of freedom for the multiplier: " << this->dof_handler.n_dofs() << endl << endl;
    }

    void change_boundary( double current_time ){
        this->constraints.clear();
        DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
        typename FunctionMap<dim>::type boundary_functions;

        Boundary<dim> boundary_solution;
        boundary_solution.set_time(current_time);

        boundary_functions[0] = &boundary_solution;
        VectorTools::project_boundary_values(this->dof_handler, boundary_functions, QGauss < dim - 1 > (this->degree + 1), this->constraints);
        this->constraints.close();
    }

    void assemble_system(bool globalProblem , const double current_time) {

        FEValues<dim>     fe_values_local(this->fe_local, this->quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values(this->fe, this->face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim> fe_face_values_local(this->fe_local, this->face_quadrature, update_values | update_gradients | update_normal_vectors | update_quadrature_points | update_JxW_values);

        const unsigned int faces_per_cell       = GeometryInfo<dim>::faces_per_cell;
        const unsigned int dofs_local_per_cell  = this->fe_local.dofs_per_cell;
        const unsigned int dofs_multiplier      = this->fe.dofs_per_cell;
        const unsigned int n_q_points           = this->quadrature.size();
        const unsigned int n_face_q_points      = this->face_quadrature.size();

        FullMatrix<double>  A_matrix            (dofs_local_per_cell, dofs_local_per_cell);
        FullMatrix<double>  B_matrix            (dofs_local_per_cell, dofs_multiplier);
        FullMatrix<double>  BT_matrix           (dofs_multiplier, dofs_local_per_cell);
        FullMatrix<double>  C_matrix            (dofs_multiplier, dofs_multiplier);
        Vector<double>      F_vector            (dofs_local_per_cell);
        Vector<double>      U_vector            (dofs_local_per_cell);
        Vector<double>      G_vector            (dofs_multiplier);

        vector<double>          st_values               ( n_q_points        );
        vector<double>          prev_solution           ( n_q_points        );
        vector<double>          mult_values             ( n_face_q_points   );
        vector<Tensor<1,dim> >  local_velocities        ( n_q_points        );
        vector<Tensor<1,dim> >  local_face_velocities   ( n_face_q_points   );

        source_term.set_time( current_time );
        const FEValuesExtractors::Vector diff_flow(0);
        const FEValuesExtractors::Scalar concentration(dim);

//        vector<double>          prev_mult               ( n_face_q_points );
//        vector<double>          prev_c                  ( n_q_points );
//        vector<Tensor<1,dim> >  prev_c_grad             ( n_q_points );
//        vector<Tensor<1,dim> >  prev_sigma              ( n_q_points );
//        vector<double>          prev_sigma_div          ( n_q_points );

        Tensor<1, dim> u_vel;               // Velocity Vector (Field)
        Tensor<2, dim> D;                   // Dispersion Tensor
        for( int i = 0; i < dim; i++) {
            u_vel[i] = coef_conv;
            D[i][i] = epsilon;
        }

        const double delta1 = -0.5; // 5*h_mesh; // TODO: Other deltas?
        const double delta2 =  0.;
        const double delta3 =  0.;

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {

            typename DoFHandler<dim>::active_cell_iterator loc_cell (&this->triangulation, cell->level(), cell->index(), &this->dof_handler_local);
            fe_values_local.reinit(loc_cell);

            A_matrix    = 0.;
            F_vector    = 0.;
            B_matrix    = 0.;
            BT_matrix   = 0.;
            C_matrix    = 0.;
            U_vector    = 0.;
            G_vector    = 0.;

            source_term.value_list(fe_values_local.get_quadrature_points(), st_values);
            fe_values_local[concentration].get_function_values( this->solution_local , prev_solution);

            for (unsigned int q = 0; q < n_q_points; ++q) {

                const double JxW = fe_values_local.JxW(q);
                for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                    const Tensor<1, dim> phi_i_s = fe_values_local[diff_flow].value(i, q);
                    const double div_phi_i_s = fe_values_local[diff_flow].divergence(i, q);
                    const double phi_i_c = fe_values_local[concentration].value(i, q);
                    const Tensor<1, dim> grad_phi_i_c = fe_values_local[concentration].gradient(i, q);
                    auto curl_phi_i_s = fe_values_local[diff_flow].curl(i, q);

                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                        const Tensor<1, dim> phi_j_s = fe_values_local[diff_flow].value(j, q);
                        const double div_phi_j_s = fe_values_local[diff_flow].divergence(j, q);
                        const double phi_j_c = fe_values_local[concentration].value(j, q);
                        const Tensor<1, dim> grad_phi_j_c = fe_values_local[concentration].gradient(j, q);
                        auto curl_phi_j_s = fe_values_local[diff_flow].curl(j, q);

                        A_matrix(i,j) +=((invert(D) * phi_j_s *      phi_i_s  // D      < sigma     , v      >
                                         -            phi_j_c *  div_phi_i_s  //        < c         , div(v) >
                                         -        div_phi_j_s *      phi_i_c  //        < div(sigma), k      >
//                                         - u_vel*grad_phi_j_c *      phi_i_c  // Oikawa
                                         + u_vel *    phi_j_c * grad_phi_i_c  /// Egger partial integration
                                         ) * dt - porosity * phi_j_c *      phi_i_c  // phi/dt < c         , q      >
                                         ) * JxW;

                        A_matrix(i, j) += delta1 * dt * D * ( invert(D) * phi_j_s + grad_phi_j_c)
                                          * ( invert(D) * phi_i_s + grad_phi_i_c) * JxW;

                        A_matrix(i, j) += delta2 * linfty_norm(invert(D))
                                          * (( div_phi_j_s + u_vel * grad_phi_j_c ) * dt + porosity * phi_j_c)
                                                                                          *  div_phi_i_s * JxW;

                        A_matrix(i, j) += delta3 * dt * ( linfty_norm(D) * curl_phi_i_s * curl_phi_j_s ) * JxW;
                    }
                    F_vector(i) +=  ( st_values[q] * dt + porosity * prev_solution[q])
                                    * (-phi_i_c + delta2 * linfty_norm(invert(D)) * div_phi_i_s) * JxW;
                }
            }
            for (unsigned int face_n = 0; face_n < faces_per_cell; ++face_n) {
                fe_face_values.reinit(cell, face_n);
                fe_face_values_local.reinit(loc_cell, face_n);

                for (unsigned int q = 0; q < n_face_q_points; ++q) {
                    const double JxW = fe_face_values_local.JxW(q);
                    const Tensor<1, dim> normal = fe_face_values_local.normal_vector(q);
                    for (unsigned int i = 0; i < dofs_local_per_cell; ++i){
                        const double phi_i_c = fe_face_values_local[concentration].value(i, q);
                        const Tensor<1, dim> grad_phi_i_c = fe_values_local[concentration].gradient(i, q);

                        for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                            const double phi_j_c = fe_face_values_local[concentration].value(j, q);
                            const Tensor<1, dim> grad_phi_j_c = fe_values_local[concentration].gradient(j, q);

//                            A_matrix(i, j)  += dt * phi_j_c * fmax(-u_vel * normal,0) * phi_i_c * JxW ; /// Oikawa
                            if (u_vel * normal >= 0)
                                A_matrix(i, j)  -= dt * u_vel * normal * phi_j_c * phi_i_c * JxW; /// Egger
                        }
                    }
                }
            }
            if (globalProblem) {
                for (unsigned int face_n = 0; face_n < faces_per_cell; ++face_n) {
                    fe_face_values.reinit(cell, face_n);
                    fe_face_values_local.reinit(loc_cell, face_n);

                    for (unsigned int q = 0; q < n_face_q_points; ++q) {

                        const double JxW = fe_face_values.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int j = 0; j < dofs_multiplier; ++j) {
                            const double phi_j_m = fe_face_values.shape_value(j, q);

                            for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                                const Tensor<1, dim> phi_i_s = fe_face_values_local[diff_flow].value(i, q);
                                const double phi_i_c = fe_face_values_local[concentration].value(i, q);

                                B_matrix (i, j) +=  dt * (phi_i_s * normal) * phi_j_m * JxW;
                                BT_matrix(j, i) +=  dt * (phi_i_s * normal) * phi_j_m * JxW;

//                                B_matrix (i, j) -= dt * phi_j_m * fmax(-u_vel * normal,0) * phi_i_c * JxW; /// Oikawa
//                                BT_matrix(j, i) -= dt * phi_j_m * fmax( u_vel * normal,0) * phi_i_c * JxW; /// Oikawa
                                if(u_vel * normal >= 0)
                                    BT_matrix(j, i) += dt * u_vel * normal * phi_j_m * phi_i_c * JxW; /// Egger
                                else
                                    B_matrix (i, j) -= dt * u_vel * normal * phi_j_m * phi_i_c * JxW; /// Egger
                            }
                            if(u_vel * normal < 0) /// Egger
                                for (unsigned int i = 0; i < dofs_multiplier; ++i) {
                                    const double phi_i_m = fe_face_values.shape_value(i, q);
                                    C_matrix(i, j) -= dt * u_vel * normal * phi_j_m * phi_i_m * JxW; /// Egger
//                                    C_matrix(i, j) -= dt * phi_j_m * fmax(u_vel * normal,0) * phi_i_m * JxW; /// Oikawa
                                }
                        }
                    }
                }
                this->static_condensation(cell,A_matrix,B_matrix,BT_matrix,C_matrix,F_vector,G_vector);
            } else {
                for (unsigned int face_n = 0; face_n < faces_per_cell; ++face_n) {
                    fe_face_values.reinit(cell, face_n);
                    fe_face_values_local.reinit(loc_cell, face_n);
                    fe_face_values.get_function_values( this->solution, mult_values);

                    for (unsigned int q = 0; q < n_face_q_points; ++q) {
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values_local.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_s = fe_face_values_local[diff_flow].value(i, q);
                            const double phi_i_c = fe_face_values_local[concentration].value(i, q);

                            F_vector(i) -= dt * phi_i_s * normal * mult_values[q] * JxW;

//                            F_vector(i) += dt * fmax(-u_vel * normal,0) * phi_i_c * mult_values[q] * JxW; /// Oikawa
                            if(u_vel * normal <= 0)
                                F_vector(i) += dt * u_vel * normal * phi_i_c * mult_values[q] * JxW;  /// Egger

                        }
                    }
                }
                A_matrix.gauss_jordan();                                // A = A^{-1}
                A_matrix.vmult(U_vector, F_vector, false);    // A^{-1} * F
//                loc_cell->set_dof_values(U_vector, this->solution_local);       // using dt = h^{k+1}
                loc_cell->set_dof_values(U_vector, this->solution_local_mid);   // using dt = h^{(k+1)/2}/2
            }
        }
    }

    void compute_errors_finaltime(ConvergenceTable &convergence_table, double ftime ) {

        ExactSolution<dim> exact_sol;
        exact_sol.set_time(ftime);

        const ComponentSelectFunction<dim> scalar_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> vector_mask(make_pair(0, dim), dim + 1);

        Vector<double> cellwise_errors(this->triangulation.n_active_cells());

        double s_l2_error, c_l2_error;

        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local, exact_sol, cellwise_errors, this->quadrature, VectorTools::L2_norm, &scalar_mask);
        c_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

//        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local, exact_sol, cellwise_errors, this->quadrature, VectorTools::L2_norm, &vector_mask);

        exact_sol.set_time(ftime - dt);
        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local_mid, exact_sol, cellwise_errors, this->quadrature, VectorTools::L2_norm, &vector_mask);
        s_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

        convergence_table.add_value("cells", this->triangulation.n_active_cells());
        convergence_table.add_value("L2_c", c_l2_error);
        convergence_table.add_value("L2_sigma", s_l2_error);
    }
};

int main () {
    const int    dim        = 2;
    const int    degree     = 2;
    const double final_time = pow( 0.25, (degree+1.0)/2 );

    ConvergenceTable convergence_table;
    for (int refinement = 2; refinement <= 6; ++refinement) {
//        double final_time = pow( pow(2.0, -refinement ), (degree+1.0)/2 ); /// Just one iteration each mesh
        cout << " --- Refinement #" << refinement << " --- " << endl;
        MixedTransport<dim> mixedTransp(degree);
        mixedTransp.run(refinement, convergence_table, final_time);
    }
    show_convergence(convergence_table , dim, "Trans_DGQ_D" );
}