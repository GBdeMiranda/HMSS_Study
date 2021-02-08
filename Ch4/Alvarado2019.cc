/*
 * Author: Gabriel Brandao de Miranda
 * Date: 28/06/19.
 */

#include <iostream>
#include <cmath>
#include <deal.II/dofs/dof_tools.h>
#include "MixedTransportBase.h"

double epsilon          = 1e-6;
double porosity         = 1.0;
const double coef_conv  = 1.0;

/// TEST THE BOUNDARY FROM OIKAWA WITH EXP(-t) MULTIPLYING
//template <int dim>
//class SourceTerm : public Function<dim> {
//public:
//
//    SourceTerm() : Function<dim>() { }
//    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
////        double return_value = 0.0;
////
////        for (int i = 0; i < dim; ++i) {
////            double laplacian  = (M_PI*M_PI/4 - 1/pow(epsilon,2)) * sin(M_PI*p[i]/2)  * exp( (p[i]-1)/epsilon )
////                                - (M_PI/epsilon)                   * cos(M_PI*p[i]/2)  * exp( (p[i]-1)/epsilon )
////                                - (M_PI*M_PI/4)                    * sin(M_PI*p[i]/2);
////            for (unsigned int j = 0; j < dim; ++j) {
////                if (i != j) {
////                    laplacian *= sin(M_PI*p[j]/2) * (1 - exp( (p[j]-1)/epsilon ) );
////                }
////            }
////            return_value -= epsilon * exp( -this->get_time() ) * laplacian;
////        }
////        for (unsigned int i = 0; i < dim; ++i) {
////            double gradient = exp( -this->get_time() );
////            for (unsigned int j = 0; j < dim; ++j) {
////                if (i == j) {
////                    gradient *= (M_PI/2)    * cos( M_PI*p[j]/2 ) * (1 - exp( (p[j]-1)/epsilon )  )
////                                -   (1/epsilon) * sin( M_PI*p[j]/2 ) *      exp( (p[j]-1)/epsilon )  ;
////                } else {
////                    gradient *= sin(M_PI * p[j]/2) * ( 1 - exp( (p[j]-1)/epsilon )  ) ;
////                }
////            }
////            return_value += gradient;
////        }
////
////        double exact = exp( -this->get_time() );
////        for (int i = 0; i < dim; ++i) {
////            exact *= sin( M_PI*p[i]/2) * (1 - exp( (p[i]-1)/epsilon ) );
////        }
////        return_value -= exact;
////        return return_value;
//
//        return -epsilon*(-1.0/4.0*pow(M_PI, 2)*(1 - exp(-1/epsilon))*(1 - exp((p[0] - 1)/epsilon))
//                         *exp(-this->get_time())*sin((1.0/2.0)*M_PI*p[0])*sin((1.0/2.0)*M_PI*p[1]) - M_PI*(1 - exp(-1/epsilon))
//                                                                                                     *exp(-this->get_time())*exp((p[0] - 1)/epsilon)*sin((1.0/2.0)*M_PI*p[1])*cos((1.0/2.0)*M_PI*p[0])/epsilon
//                         - (1 - exp(-1/epsilon))*exp(-this->get_time())*exp((p[0] - 1)/epsilon)*sin((1.0/2.0)*M_PI*p[0])
//                           *sin((1.0/2.0)*M_PI*p[1])/pow(epsilon, 2)) - (1 - exp(-1/epsilon))*(1 - exp((p[0] - 1)/epsilon))
//                                                                        *exp(-this->get_time())*sin((1.0/2.0)*M_PI*p[0])*sin((1.0/2.0)*M_PI*p[1]) + (1.0/4.0)*pow(M_PI, 2)
//                                                                                                                                                    *(1 - exp(-1/epsilon))*(1 - exp((p[0] - 1)/epsilon))
//                                                                                                                                                    *exp(-this->get_time())*sin((1.0/2.0)*M_PI*p[0])*sin((1.0/2.0)*M_PI*p[1]) + (1.0/2.0)
//                                                                                                                                                                                                                                *M_PI*(1 - exp(-1/epsilon))*(1 - exp((p[0] - 1)/epsilon))*exp(-this->get_time())
//                                                                                                                                                                                                                                *sin((1.0/2.0)*M_PI*p[0])*cos((1.0/2.0)*M_PI*p[1]) + (1.0/2.0)*M_PI*(1 - exp(-1/epsilon))
//                                                                                                                                                                                                                                                                                     *(1 - exp((p[0] - 1)/epsilon))*exp(-this->get_time())*sin((1.0/2.0)*M_PI*p[1])
//                                                                                                                                                                                                                                                                                     *cos((1.0/2.0)*M_PI*p[0]) - (1 - exp(-1/epsilon))*exp(-this->get_time())
//                                                                                                                                                                                                                                                                                                                 *exp((p[0] - 1)/epsilon)*sin((1.0/2.0)*M_PI*p[0])*sin((1.0/2.0)*M_PI*p[1])/epsilon;
//    }
//};
//
//template <int dim>
//class ExactSolution : public Function<dim> {
//public: ExactSolution() : Function<dim>(dim+1) { }
//    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
//        Assert (values.size() == dim+1, ExcDimensionMismatch (values.size(), dim+1));
//
//        for (unsigned int i = 0; i < dim; ++i) {
//            values[i] = - epsilon * exp( -this->get_time() );
//            for (unsigned int j = 0; j < dim; ++j) {
//                if (i == j) {
//                    values[i] *= (M_PI/2)    * cos( M_PI*p[j]/2 ) * (1 - exp( (p[j]-1)/epsilon )  )
//                                 -   (1/epsilon) * sin( M_PI*p[j]/2 ) *      exp( (p[j]-1)/epsilon )  ;
//                } else {
//                    values[i] *= sin(M_PI * p[j]/2) * ( 1 - exp( (p[j]-1)/epsilon )  ) ;
//                }
//            }
//        }
//        values[dim] = exp( -this->get_time() );
//        for (int i = 0; i < dim; ++i) {
//            values[dim] *= sin( M_PI*p[i]/2) * (1 - exp( (p[i]-1)/epsilon ) );
//        }
//
//    }
//};
//
//template <int dim>
//class Boundary : public Function<dim> {
//public: Boundary() : Function<dim>() { }
//    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
//        double return_value = exp( -this->get_time() );
//
//        for (int i = 0; i < dim; ++i) {
//            return_value *= sin( M_PI*p[i]/2) * (1 - exp( (p[i]-1)/epsilon ) );
//        }
//        return return_value;
//    }
//};
//
//template <int dim>
//class InitialCondition : public Function<dim> {
//public: InitialCondition() : Function<dim>(dim+1) { }
//    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
//        for (unsigned int i = 0; i < dim; ++i) {
//            values[i] = - epsilon;
//            for (unsigned int j = 0; j < dim; ++j) {
//                if (i == j) {
//                    values[i] *= (M_PI/2)    * cos( M_PI*p[j]/2 ) * (1 - exp( (p[j]-1)/epsilon )  )
//                                 -   (1/epsilon) * sin( M_PI*p[j]/2 ) *      exp( (p[j]-1)/epsilon )  ;
//                } else {
//                    values[i] *= sin(M_PI * p[j]/2) * ( 1 - exp( (p[j]-1)/epsilon )  ) ;
//                }
//            }
//        }
//        values[dim] = 1.0;
//        for (int i = 0; i < dim; ++i) {
//            values[dim] *= sin( M_PI*p[i]/2) * (1 - exp( (p[i]-1)/epsilon ) );
//        }
//    }

template <int dim>
class SourceTerm : public Function<dim> {
public: SourceTerm() : Function<dim>() { }
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {

///  2D Source from python sympy
        return (1 - exp(-p[0]/epsilon))*(1 - exp(-p[1]/epsilon))*exp(this->get_time())/pow(1 - exp(-1/epsilon), 2)
        + (1 - exp(-p[0]/epsilon))*exp(this->get_time())*exp(-p[1]/epsilon)/(epsilon*pow(1 - exp(-1/epsilon), 2))
        + 2*(1 - exp(-p[1]/epsilon))*exp(this->get_time())*exp(-p[0]/epsilon)/(epsilon*pow(1 - exp(-1/epsilon), 2))
        + (1 - exp(-p[0]/epsilon))*exp(this->get_time())*exp(-p[1]/epsilon)/(pow(epsilon, 2)*pow(1 - exp(-1/epsilon), 2));

        return  (1 - exp(-p[0]/epsilon))*(1 - exp(-p[1]/epsilon))*exp(this->get_time())/pow(1 - exp(-1/epsilon), 2)
                + (1 - exp(-p[0]/epsilon))*exp(this->get_time())*exp(-p[1]/epsilon)/( epsilon*pow(1 - exp(-1/epsilon), 2))
                + (1 - exp(-p[1]/epsilon))*exp(this->get_time())*exp(-p[0]/epsilon)/( epsilon*pow(1 - exp(-1/epsilon), 2))
                + (1 - exp(-p[0]/epsilon))*exp(this->get_time())*exp(-p[1]/epsilon)/(epsilon*pow(1 - exp(-1/epsilon), 2))
                + (1 - exp(-p[1]/epsilon))*exp(this->get_time())*exp(-p[0]/epsilon)/(epsilon*pow(1 - exp(-1/epsilon), 2));

        double dct          = exp( this->get_time() ) / pow( 1-exp(-1/epsilon),  dim );
        double dot_grad     = 0;
        double laplacian    = 0;

        for(int i = 0; i < dim; i++){
            dct *= 1 - exp(-p[i]/epsilon);

            double dcx_i = exp(-p[i]/epsilon);  // aux for dot_grad and laplacian
            for(int j = 0; j < dim; j++){
                if(i != j)  dcx_i *= 1 - exp(-p[j]/epsilon);
            }
            dot_grad  += coef_conv * dcx_i;
            laplacian += dcx_i;
        }

        dot_grad  *= exp( this->get_time() ) / (     epsilon    * pow(1-exp(-1/epsilon), dim) );
        laplacian *= exp( this->get_time() ) / ( pow(epsilon,2) * pow(1-exp(-1/epsilon), dim) );

        return dct + dot_grad - laplacian;

    }
};

template <int dim>
class InitialCondition : public Function<dim> {
public: InitialCondition() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        values[dim] = 1 / pow(  1 - exp(-1/epsilon), dim );
        for(int i = 0; i < dim; i++) {
            values[dim] *= ( 1 - exp(-p[i]/epsilon) );

            values[i] = - exp(- p[i]/epsilon) ;
            for(int j = 0; j < dim; j++) {
                if( i != j)
                    values[i] *= 1-exp(-p[j]/epsilon) ;
            }
        }
    }
};

template <int dim>
class ExactSolution : public Function<dim> {
public: ExactSolution() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        Assert (values.size() == dim+1, ExcDimensionMismatch (values.size(), dim+1));

        values[dim] = exp( this->get_time() ) / pow( 1-exp(-1/epsilon ), dim );
        for(int i = 0; i < dim; i++) {
            values[dim] *= ( 1 - exp(-p[i]/epsilon) );

            values[i] = -exp( this->get_time() - p[i]/epsilon ) / pow( 1-exp(-1/epsilon ), dim);
            for(int j = 0; j < dim; j++) {
                if( i != j)
                   values[i] *= ( 1-exp(-p[j]/epsilon) );
            }
        }
    }
};

template <int dim>
class Boundary : public Function<dim> {
public: Boundary() : Function<dim>() { }
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        double return_value = exp( this->get_time() ) / pow( 1 - exp(-1/epsilon), dim );
        for(int i = 0; i < dim; i++)
            return_value *= 1 - exp(-p[i]/epsilon);

        return return_value;
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

        int step = 0;
        this->output_results( refinements, step, "Alvarado/plots/" );
        while( current_time < final_time ) {
            cout << "   -> Current time:  " << current_time << endl << endl;
            current_time += dt;

            change_boundary(current_time);
            assemble_system( true, current_time);
            this->solve();
            assemble_system( false, current_time );

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);

            for(int i = 0; i < this->dof_handler_local.n_dofs(); i++)
                this->solution_local[i] = 2 * this->solution_local_mid[i] - this->solution_local[i];
            current_time += dt; // using dt = h^{(k+1)/2}/2

            step += 1;
            this->output_results( refinements, step, "Alvarado/plots/" );
        }
        cout << "   -> Current time:  " << current_time << endl << endl;
        compute_errors_subdomain(convergence_table, current_time );
    }

private:
    double h_mesh;
    double dt;

    SourceTerm<dim> source_term;

    void make_grid_and_dofs( int numberCells, double inf_domain, double sup_domain ) {
        GridGenerator::hyper_cube(this->triangulation, inf_domain, sup_domain);
        this->triangulation.refine_global(numberCells);

        this->dof_handler.distribute_dofs(this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        h_mesh = GridTools::maximal_cell_diameter(this->triangulation)*sqrt(2)/2;
        dt = pow( h_mesh, (this->degree + 1.0)/2 )/2;
//        dt = pow( h_mesh, this->degree + 1 );

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

        Tensor<1, dim> u_vel;               // Velocity Vector (Field)
        Tensor<2, dim> D;                   // Dispersion Tensor
        for( int i = 0; i < dim; i++) {
            u_vel[i] = coef_conv;
        }
        D[0][0] = epsilon;
        D[1][1] = 1.0;

        const double delta1 = -0.; // 5*h_mesh; // TODO: Other deltas?
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
//                loc_cell->set_dof_values( U_vector, this->solution_local );
                loc_cell->set_dof_values( U_vector, this->solution_local_mid ); // using dt = h^{(k+1)/2}/2
            }
        }
    }
    void compute_errors_subdomain(ConvergenceTable &convergence_table, double ftime ) {
        ExactSolution<dim> exact_sol;

        const ComponentSelectFunction<dim> scalar_mask(dim, dim + 1);
        const ComponentSelectFunction<dim> vector_mask(std::make_pair(0, dim), dim + 1);

        Vector<double> cellwise_errors(this->triangulation.n_active_cells());

        QGauss<dim> quadrature(this->degree + 2);
        double s_l2_error, c_l2_error;

        double limit = 0.8;     // Percent of the domain where the error is evaluated

        exact_sol.set_time(ftime - dt);
        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local_mid, exact_sol, cellwise_errors, quadrature, VectorTools::L2_norm, &vector_mask);
//        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local, exact_sol, cellwise_errors, quadrature, VectorTools::L2_norm, &vector_mask);
        for (const auto &c : this->dof_handler.active_cell_iterators())
            if (c->center()[0] > limit || c->center()[1] > limit ) cellwise_errors(c->index()) = 0.0;
        s_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

        exact_sol.set_time(ftime);
        VectorTools::integrate_difference(this->dof_handler_local, this->solution_local, exact_sol, cellwise_errors, quadrature, VectorTools::L2_norm, &scalar_mask);
        for (const auto &c : this->dof_handler.active_cell_iterators())
            if (c->center()[0] > limit || c->center()[1] > limit ) cellwise_errors(c->index()) = 0.0;
        c_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

        convergence_table.add_value("cells", this->triangulation.n_active_cells());
        convergence_table.add_value("L2_c", c_l2_error);
        convergence_table.add_value("L2_sigma", s_l2_error);
    }
};

int main () {
    const int    dim        = 2;
    const int    degree     = 1;
    const double final_time = pow( 0.25, (degree+1.0)/2 );

//    for (int iter = 0; iter < 3; iter++) {

        ConvergenceTable convergence_table;
        for (int refinement = 2; refinement <= 6; ++refinement) {
            cout << " --- Refinement #" << refinement << " --- " << endl;
            MixedTransport<dim> mixedTransp(degree);
            mixedTransp.run(refinement, convergence_table, final_time);
        }
        show_convergence(convergence_table , dim, "Alvarado_RT_C_" );// + Utilities::to_string(iter, 0) );

//        epsilon *= 1e-3;
//    }
}