
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include "UnifiedBase.h"

using namespace dealii;
using namespace std;

const double epsilon = 1;
const double fi = 1.0 , permeability = -1.0;

template <int dim>
class SourceTerm : public Function<dim> {
public: SourceTerm() : Function<dim>() { }
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return 2 * pow(M_PI,2) * epsilon * sin( M_PI * ( p[0] + p[1] - 2*this->get_time() ) );
    }
};

template <int dim>
class InitialCondition : public Function<dim> {
public: InitialCondition() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        for(int i = 0; i < dim; i++) values[i] = - epsilon * M_PI * cos( M_PI * ( p[0] + p[1] - 2*this->get_time() ) );
        values[dim] = sin( M_PI * ( p[0] + p[1] - 2*this->get_time() ) );
    }
};

template<int dim> class PressureBoundary : public Function<dim> {
public: PressureBoundary() : Function<dim>(1) {}
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return (0.2/M_PI) * cos( M_PI*( p[0] + p[1] - 2*this->get_time() ) ) + 0.5*( p[0] + p[1] );
    }
};

template<int dim> class ConcentrationBoundary : public Function<dim> {
public: ConcentrationBoundary() : Function<dim>(1) {}
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {

        return sin( M_PI * ( p[0] + p[1] - 2*this->get_time() ) );
    }
};

template<int dim>
class ExactSolutionDarcy : public Function<dim> {
public: ExactSolutionDarcy() : Function<dim>(dim + 1) {}
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));

        for (unsigned int i = 0; i < dim; ++i)
            values[i] = 1.0;

        values[dim] = (0.2/M_PI) * cos( M_PI*( p[0] + p[1] - 2*this->get_time() ) ) + 0.5*( p[0] + p[1] );
    }
};

template<int dim>
class ExactSolutionTransport : public Function<dim> {
public: ExactSolutionTransport() : Function<dim>(dim + 1) {}
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        Assert(values.size() == dim + 1, ExcDimensionMismatch(values.size(), dim + 1));

        for (unsigned int i = 0; i < dim; ++i)
            values[i] = - M_PI * epsilon * cos( M_PI*( p[0] + p[1] - 2*this->get_time() ) ) ;

        values[dim] = sin( M_PI * ( p[0] + p[1] - 2*this->get_time() ) );
    }
};

template <int dim>
class UnifiedProblem : UnifiedBase<dim>{
public:
    UnifiedProblem (const unsigned int degree) : UnifiedBase<dim>(degree) {}

    void run (int numberCells, double inf_domain, double sup_domain, const double final_time , ConvergenceTable &convergence_table ) {

        make_grid_and_dofs( numberCells, inf_domain, sup_domain);

        InitialCondition<dim> initial_condition;
        this->constraints.close();
        VectorTools::project( this->dof_handler_local, this->constraints, QGauss<dim>(this->degree+2), initial_condition, this->transp_solution_local );

        double current_time  = 0; // dt;
        cout << "   -> Current time:  " << current_time << endl;

        change_boundary_darcy(current_time);

        //Darcy -- u^n
        assemble_system (true, false, current_time);
        this->solve ();
        assemble_system (false, false, current_time);
        this->system_rhs.reinit(this->dof_handler.n_dofs());
        this->system_matrix.reinit(this->sparsity_pattern);

        while( current_time < final_time ) {
            current_time += dt;
            //Transport -- c^{n+1/2}
            change_boundary_transport(current_time);
            assemble_system (true, true, current_time);
            this->solve ();
            assemble_system (false, true, current_time);

            current_time += dt;
            //Transport -- c^{n+1} = 2c^{n+1/2} - c^{n}
            this->transp_solution_local.add(2. , this->transp_solution_local_mid, -2., this->transp_solution_local); // c^{n} = c^{n} + 2c^{n+1/2} - 2c^{n}

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);

            //Darcy -- u^{n+1}
            change_boundary_darcy(current_time);

            assemble_system (true, false, current_time);
            this->solve();
            assemble_system (false, false, current_time);

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);

            cout << "   -> Current time:  " << current_time << endl;

        }
//        this->output_results (true);
//        this->output_results (false);

        ExactSolutionDarcy<dim> exact_solution_darcy;
        ExactSolutionTransport<dim> exact_solution_transp;
        exact_solution_darcy.set_time(current_time);
        exact_solution_transp.set_time(current_time);
        this->compute_errors( convergence_table, exact_solution_darcy, exact_solution_transp );
    }

private:
    double h_mesh;
    double dt;

    SourceTerm<dim> source_term;

    void change_boundary_transport( double current_time  ){
            ConcentrationBoundary<dim> boundary_function;
            boundary_function.set_time(current_time);
            this->constraints.clear();
            DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
            VectorTools::interpolate_boundary_values(this->dof_handler, 0, boundary_function, this->constraints);
            this->constraints.close();
    }
    void change_boundary_darcy( double current_time  ){
            PressureBoundary<dim> boundary_function;
            boundary_function.set_time(current_time);
            this->constraints.clear();
            DoFTools::make_hanging_node_constraints(this->dof_handler, this->constraints);
            VectorTools::interpolate_boundary_values(this->dof_handler, 0 , boundary_function, this->constraints );
            this->constraints.close();
    }

    void make_grid_and_dofs ( int numberCells, double inf_domain, double sup_domain ) {

        GridGenerator::hyper_cube(this->triangulation, inf_domain, sup_domain);
        this->triangulation.refine_global(numberCells);

        this->dof_handler.distribute_dofs (this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        h_mesh = GridTools::maximal_cell_diameter(this->triangulation)*sqrt(2.)/2.;
//        dt = pow( h_mesh, this->degree + 1 );
        dt = pow( h_mesh, (this->degree + 1.0)/2 )/2.;

        /// Set all boundary dirichlet
        for ( const auto &cell : this->triangulation.cell_iterators() )
            for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
                if (cell->face(face)->at_boundary() )
                    cell->face(face)->set_boundary_id( 0 );

        this->solution.reinit (this->dof_handler.n_dofs());
        this->system_rhs.reinit (this->dof_handler.n_dofs());
        this->transp_solution_local.reinit (this->dof_handler_local.n_dofs());
        this->transp_solution_local_mid.reinit (this->dof_handler_local.n_dofs());
        this->darcy_solution_local.reinit (this->dof_handler_local.n_dofs());


        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (this->dof_handler, dsp, this->constraints, false);
        this->sparsity_pattern.copy_from(dsp);
        this->system_matrix.reinit(this->sparsity_pattern);

        this->constraints.close ();
    }

    void assemble_system (bool globalProblem, bool Transport, const double current_time ) {

        FEValues<dim>       fe_values_local(this->fe_local, this->quadrature, update_values | update_gradients      | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim>   fe_face_values (this->fe,  this->face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);
        FEFaceValues<dim>   fe_face_values_local (this->fe_local, this->face_quadrature, update_values | update_normal_vectors | update_quadrature_points | update_JxW_values);

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
        
        vector<double>      mult_values         ( n_face_q_points );
        vector<double>      st_values           ( n_q_points );
        vector<double>      concentration       ( n_q_points );

        vector<Tensor<1,dim> >  velocities      ( n_q_points, Tensor<1,dim>({1,1}) );
        vector<Tensor<1,dim> >  face_vels ( n_face_q_points);

        source_term.set_time( current_time );
        const FEValuesExtractors::Vector  vector_var (0);
        const FEValuesExtractors::Scalar  scalar_var (dim);

        double k_inverse = 1/epsilon;

        const double delta1 = -0.5, delta2 = 0.5, delta3 = 0.5;

//        const double oikawa = -1.0; //Oikawa
        double tr;  /// Reduce the conditionals in loops
        if(Transport) {
            tr = 1;
        } else {
            tr = 0;
        }
        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            typename DoFHandler<dim>::active_cell_iterator loc_cell (&this->triangulation, cell->level(), cell->index(), &this->dof_handler_local);
            fe_values_local.reinit (loc_cell);

            A_matrix  = 0;
            F_vector  = 0;
            B_matrix  = 0;
            BT_matrix = 0;
            U_vector  = 0;
            C_matrix  = 0;
            G_vector  = 0;
            
            source_term.value_list( fe_values_local.get_quadrature_points(), st_values);

            fe_values_local[scalar_var].get_function_values( this->transp_solution_local , concentration);
            fe_values_local[vector_var].get_function_values( this->darcy_solution_local , velocities);

            for (unsigned int q=0; q<n_q_points; ++q) {

                double viscosity;
                if(!Transport){
                    viscosity = 0.5 - 0.2 * concentration[q] ;
                }

                const double JxW = fe_values_local.JxW(q);
                for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                    const Tensor<1, dim> phi_i_s = fe_values_local[vector_var].value(i, q);
                    const double div_phi_i_s = fe_values_local[vector_var].divergence(i, q);
                    auto curl_phi_i_s = fe_values_local[vector_var].curl(i, q);
                    const double phi_i_c = fe_values_local[scalar_var].value(i, q);
                    const Tensor<1, dim> grad_phi_i_c = fe_values_local[scalar_var].gradient(i, q);

                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                        const Tensor<1, dim> phi_j_s = fe_values_local[vector_var].value(j, q);
                        const double div_phi_j_s = fe_values_local[vector_var].divergence(j, q);
                        auto curl_phi_j_s = fe_values_local[vector_var].curl(j, q);
                        const double phi_j_c = fe_values_local[scalar_var].value(j, q);
                        const Tensor<1, dim> grad_phi_j_c = fe_values_local[scalar_var].gradient(j, q);

                        if(Transport){
                            A_matrix(i, j) += (dt * (  k_inverse  *      phi_j_s *      phi_i_s
                                                    -                      phi_j_c *  div_phi_i_s
                                                    -                  div_phi_j_s *      phi_i_c
//                                                    - velocities[q] * grad_phi_j_c *      phi_i_c // Oikawa
                                                    + velocities[q] *      phi_j_c * grad_phi_i_c // Egger
                                                    )               - fi * phi_j_c *      phi_i_c
                                              ) * JxW;

                            A_matrix(i, j) += delta1*dt*epsilon * ( k_inverse * phi_j_s + grad_phi_j_c)
                                                                * ( k_inverse * phi_i_s + grad_phi_i_c) * JxW;
                        }else{
                            A_matrix(i, j) +=       ( (viscosity/permeability) * phi_j_s * phi_i_s
                                                     -                 phi_j_c * div_phi_i_s
                                                     -   div_phi_j_s * phi_i_c
                                                     ) * JxW;

                            A_matrix(i, j) += delta1 * (permeability/viscosity) * ((viscosity/permeability) * phi_j_s + grad_phi_j_c)
                                                                                * ((viscosity/permeability) * phi_i_s + grad_phi_i_c) * JxW;
                            A_matrix(i, j) += delta2 * (permeability/viscosity) * div_phi_i_s * div_phi_j_s * JxW;  // d2 * |A| * <div u, div v>
                            A_matrix(i, j) += delta3 * (permeability/viscosity) *curl_phi_i_s *curl_phi_j_s * JxW;  // d3 * |A| * <rot u, rot v>
                         }
                    }
                    if(Transport){
                            F_vector(i) -= (dt * st_values[q]  + fi * concentration[q]) * phi_i_c  * JxW;

                    }
                }
            }

            if(Transport) {
                /// Loop nas faces dos elementos
                for ( unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n ) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values_local[vector_var].get_function_values( this->darcy_solution_local , face_vels);

                    for ( unsigned int q = 0; q < n_face_q_points; ++q ) {
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values_local.normal_vector(q);

                        for ( unsigned int i = 0; i < dofs_local_per_cell; ++i ) {
                            const double phi_i_c = fe_face_values_local[scalar_var].value(i, q);

                            for ( unsigned int j = 0; j < dofs_local_per_cell; ++j ) {
                                const double phi_j_c = fe_face_values_local[scalar_var].value(j, q);
//                                A_matrix(i, j)  += oikawa*phi_j_c * fmax(-face_vels[q] * normal,0) * phi_i_c * dt * JxW ; // Oikawa
                                if (face_vels[q] * normal >= 0)   A_matrix(i, j)  -=  face_vels[q] * normal * phi_j_c * phi_i_c * dt * JxW; // Egger
                            }
                        }
                    }
                }
            }
            if (globalProblem) {
                for (unsigned int face_n = 0; face_n < GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values_local[vector_var].get_function_values( this->darcy_solution_local , face_vels);

                    for (unsigned int q=0; q<n_face_q_points; ++q) {
                        double JxW = ((dt-1)*tr+1)*fe_face_values_local.JxW(q);    /// Reduce the conditionals in loops
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_s = fe_face_values_local[vector_var].value(i, q);
                            const double phi_i_c = fe_face_values_local[scalar_var].value(i, q);

                            for (unsigned int j = 0; j < dofs_multiplier; ++j) {
                                const double phi_j_m = fe_face_values.shape_value(j, q);

                                    B_matrix (i, j) += phi_i_s * normal * phi_j_m * JxW;
                                    BT_matrix(j, i) += phi_i_s * normal * phi_j_m * JxW;
                                    
//                                    B_matrix (i, j) -= oikawa*phi_j_m * fmax(-face_vels[q] * normal,0) * phi_i_c * JxW;                       // Oikawa
//                                    BT_matrix(j, i) -= oikawa*phi_j_m * fmax( face_vels[q] * normal,0) * phi_i_c * JxW;                         // Oikawa

                                    if(face_vels[q] * normal < 0)   B_matrix(i, j)  -= tr * face_vels[q] * normal * phi_j_m * phi_i_c * JxW; // Egger
                                    else                            BT_matrix(j, i) += tr * face_vels[q] * normal * phi_j_m * phi_i_c * JxW; // Egger

                            }
                        }
                        if(Transport) {
                            for (unsigned int i = 0; i < dofs_multiplier; ++i) {
                                const double phi_i_m = fe_face_values.shape_value(i, q);
                                for (unsigned int j = 0; j < dofs_multiplier; ++j) {
                                    const double phi_j_m = fe_face_values.shape_value(j, q);
//                                    C_matrix(i, j) -= oikawa * phi_j_m * fmax(face_vels[q] * normal,0) * phi_i_m * dt * JxW;                         // Oikawa
                                    if(face_vels[q] * normal < 0)     C_matrix(i, j) -= face_vels[q] * normal * phi_j_m * phi_i_m  * JxW; // Egger
                                }
                            }
                        }
                    }
                }
                this->static_condensation(cell,A_matrix,B_matrix,BT_matrix,C_matrix,F_vector,G_vector);
                
            } else {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values.get_function_values (this->solution, mult_values);
                    fe_face_values_local[vector_var].get_function_values( this->darcy_solution_local , face_vels);

                    for (unsigned int q=0; q<n_face_q_points; ++q) {
                        double JxW = (tr*(dt-1)+1)*fe_face_values_local.JxW(q);    /// Reduce the conditionals in loops
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_s = fe_face_values_local[vector_var].value(i, q);
                            const double phi_i_c = fe_face_values_local[scalar_var].value(i, q);

                            F_vector(i) -= phi_i_s * normal * mult_values[q] * JxW;
//                                F_vector(i) += oikawa*fmax(-face_vels[q] * normal,0) * phi_i_c * dt * mult_values[q] * JxW;    // Oikawa
                            if(face_vels[q] * normal < 0)     F_vector(i) += tr * face_vels[q] * normal * phi_i_c * mult_values[q] * JxW;  // Egger
                        }
                    }
                }
                A_matrix.gauss_jordan();
                A_matrix.vmult(U_vector, F_vector, false);

                if(Transport) {
                    loc_cell->set_dof_values(U_vector, this->transp_solution_local_mid);
//                    loc_cell->set_dof_values(U_vector, this->transp_solution_local);
                } else {
                    loc_cell->set_dof_values(U_vector, this->darcy_solution_local);
                }
            }
        }
    }
};

int main () {

    const int dim = 2, degree = 1;
    const double final_time = pow( 0.5, (degree+1.0) );

    ConvergenceTable convergence_table;
    for (int i = 2; i <= 6; ++i) {
        cout << endl << "   ---   Refinement #" << i << "   ---   " << endl;
        UnifiedProblem<dim> unified(degree);
        unified.run(i, 0.0, 1.0, final_time, convergence_table);
    }
    show_convergence( convergence_table , dim, "DGQ_D" );
    return 0;
}
