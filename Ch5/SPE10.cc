
#include <fstream>
#include <iostream>
#include <cmath>
#include <math.h>
#include "UnifiedBase.h"

using namespace dealii;
using namespace std;

const double fi = 1.0 , alpha_mol = 1.8e-5, alpha_l = 1.8e-3, alpha_t = 1.8e-4;          // TRANSP PARAMS
const double permeability = -1.0, viscosity_res = 1., viscosity_ratio  = 100.;    // DARCY  PARAMS

template <int dim>
class InitialCondition : public Function<dim> {
public: InitialCondition() : Function<dim>(dim+1) { }
    virtual void vector_value (const Point<dim> &p, Vector<double>   &values) const {
        for(int i = 0; i <= dim; i++)
            values[i] = 0;
    }
};

template<int dim> class PressureBoundaryIn : public Function<dim> {
public: PressureBoundaryIn() : Function<dim>(1) {}
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return 1.0;
    }
};

template<int dim> class PressureBoundaryOut : public Function<dim> {
public: PressureBoundaryOut() : Function<dim>(1) {}
    virtual double value(const Point<dim> &p, const unsigned int component = 0) const {
        return 0.0;
    }
};

template<int dim> class ConcentrationBoundary : public Function<dim> {
public: ConcentrationBoundary() : Function<dim>(1) {}
    virtual double value(const Point<dim> &p, const unsigned int /*component*/ = 0) const {
            return 1.0;
    }
};

template <int dim>
class UnifiedProblem : UnifiedBase<dim>{
public:
    UnifiedProblem (const unsigned int degree) : UnifiedBase<dim>(degree) {}

    void run ( double inf_domain, double sup_domain, const double final_time ) {
        string dirname  = "output/SPE10/MobAdv/d" + Utilities::to_string(this->degree) + "/";

        make_grid_and_dofs( );

        InitialCondition<dim> initial_condition;
        this->constraints.close();
        VectorTools::project( this->dof_handler_local, this->constraints, QGauss<dim>(this->degree+2), initial_condition, this->transp_solution_local );

        this->output_results (true, dirname);
        double current_time  = 0; // dt;
        cout << "   -> Current time:  " << current_time << endl;

        //Darcy -- u^n
        change_boundary_darcy( current_time );
        assemble_system (true, false, current_time);
        this->solve();
        assemble_system (false, false, current_time);
        this->system_rhs.reinit(this->dof_handler.n_dofs());
        this->system_matrix.reinit(this->sparsity_pattern);

        this->output_results (false, dirname);

        while( current_time < final_time ) {
            current_time += dt;
            //Transport -- c^{n+1/2}
            change_boundary_transport(current_time);
            assemble_system (true, true, current_time);
            this->solve();
            assemble_system (false, true, current_time);

            current_time += dt; // c^{n+1} = 2c^{n+1/2} - c^{n}

            this->transp_solution_local.add(2. , this->transp_solution_local_mid, -2., this->transp_solution_local); // c^{n} = c^{n} + 2c^{n+1/2} - 2c^{n}

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);

            ///Darcy -- u^{n+1}  --  BEGIN   |||   Uncomment to insert strong coupling
            change_boundary_darcy(current_time);

            assemble_system (true, false, current_time);
            this->solve();
            assemble_system (false, false, current_time);

            this->system_rhs.reinit(this->dof_handler.n_dofs());
            this->system_matrix.reinit(this->sparsity_pattern);
            ///Darcy -- u^{n+1}   --  END

            cout << "   -> Current time:  " << current_time << endl;

            string filenameT = Utilities::to_string(round( current_time/(2*dt) ),0);
            this->output_results (true, dirname, filenameT );
            this->output_results (false, dirname, filenameT);
        }
//        this->output_results (true, dirname );
//        this->output_results (false, dirname);
    }

private:
    double h_mesh;
    double dt;

    void change_boundary_transport( double current_time  ){
            ConcentrationBoundary<dim> boundary_function;
            boundary_function.set_time(current_time);
            this->constraints.clear();
            VectorTools::interpolate_boundary_values(this->dof_handler, 1 , boundary_function, this->constraints );
            this->constraints.close();
    }

    void change_boundary_darcy( double current_time  ){
            PressureBoundaryIn<dim> boundary_function_In;
            PressureBoundaryOut<dim> boundary_function_Out;
            this->constraints.clear();
            VectorTools::interpolate_boundary_values(this->dof_handler, 1 , boundary_function_In, this->constraints );
            VectorTools::interpolate_boundary_values(this->dof_handler, 2 , boundary_function_Out, this->constraints );
            this->constraints.close();
    }

    void make_grid_and_dofs () {
        vector<unsigned int> repetitions = { 220, 60 };
        GridGenerator::subdivided_hyper_rectangle(this->triangulation, repetitions,
                                                  Point<2>(0.0, 0.0),
                                                  Point<2>(3.67, 1.0));

        this->dof_handler.distribute_dofs (this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        h_mesh = GridTools::maximal_cell_diameter(this->triangulation)*sqrt(2.)/2.;
        dt = 0.01;//pow( h_mesh, this->degree + 1 );
//        dt = pow( h_mesh, (this->degree + 1.0)/2 )/2.;

        /// Set all boundary ...
        ///         SLAB BY BOUNDARY CONDITIONS
        for ( const auto &cell : this->triangulation.cell_iterators() ) {
            if ( cell->face(0)->at_boundary() )
                cell->face(0)->set_boundary_id(1);
            else if(cell->face(1)->at_boundary())
                cell->face(1)->set_boundary_id(2);
        }

        this->solution.reinit (this->dof_handler.n_dofs());
        this->system_rhs.reinit (this->dof_handler.n_dofs());
        this->transp_solution_local.reinit (this->dof_handler_local.n_dofs());
        this->transp_solution_local_mid.reinit (this->dof_handler_local.n_dofs());
        this->darcy_solution_local.reinit (this->dof_handler_local.n_dofs());


        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (this->dof_handler, dsp, this->constraints, false);
        this->sparsity_pattern.copy_from(dsp);
        this->system_matrix.reinit(this->sparsity_pattern);
    }

    Tensor<2, dim> get_local_dispersion_tensor( Tensor<1, dim> b_velocity ) {
        Tensor<2, dim> dispersion_tensor;
        for( int i = 0; i < dim; i++)
            dispersion_tensor[i][i] = alpha_mol;

        for( int ii = 0; ii < dim; ii++)
            for( int jj = 0; jj < dim; jj++)
                dispersion_tensor[ii][jj] += b_velocity.norm() * (
                            alpha_l *                      b_velocity[ii]*b_velocity[jj]/b_velocity.norm_square()
                        +   alpha_t * ( 1-fabs(ii-jj) - b_velocity[ii]*b_velocity[jj]/b_velocity.norm_square() ) );

        return dispersion_tensor;
    }

    void print_data () {
        Vector<double> solution_values(dim+1); // variavel que recebe a solução em cada vértice do elemento

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            for ( unsigned int vertex=0; vertex<GeometryInfo<dim>::vertices_per_cell; ++vertex ){
                Point<dim> p = cell->vertex( vertex ); // coordenadas dos vertices
                VectorTools::point_value(this->dof_handler_local, this->solution_local, p , solution_values); // coleta a solução do solution_local no ponto "p" e guarda em solution_values

                cout<< p << " " << solution_values[0] <<  " " << solution_values[1] <<  " " << solution_values[2] << endl;
            } //  vertex loop
        } // cell loop
    }

    vector<double> load_perm_data( string perm_filename) {
        ifstream file;
        file.open(perm_filename);

        double n_cells = this->triangulation.n_active_cells();
        vector<double> permeability (n_cells);

        if( file.is_open() ){
            double x, y;
            for( size_t cell=0; cell < n_cells; cell++)
                file >> x >> y >> permeability[cell];

            file.close();
        }
        return permeability;
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
        vector<double>      concentration       ( n_q_points );

        vector<Tensor<1,dim> >  velocities      ( n_q_points, Tensor<1,dim>({1,1}) );
        vector<Tensor<1,dim> >  face_vels ( n_face_q_points);

        const FEValuesExtractors::Vector  vector_var (0);
        const FEValuesExtractors::Scalar  scalar_var (dim);

///     DARCY  PARAMS
        vector<double> permeability (this->triangulation.n_active_cells());
        permeability = load_perm_data( "spe10_permeability.txt");

        double viscosity = 1.0;

///     TRANSP PARAMS
        Tensor<2, dim> dispersion_tensor;

///     STABILIZATION PARAMS
        const double delta1 = -0.;
        const double delta2 =  0.;
        const double delta3 =  0.;

        double Pe_global = FLT_MIN;

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            typename DoFHandler<dim>::active_cell_iterator loc_cell (&this->triangulation, cell->level(), cell->index(), &this->dof_handler_local);
            fe_values_local.reinit (loc_cell);

            double k_perm = permeability[loc_cell->index()];
            A_matrix  = 0;
            F_vector  = 0;
            B_matrix  = 0;
            BT_matrix = 0;
            U_vector  = 0;
            C_matrix  = 0;
            G_vector  = 0;


            fe_values_local[scalar_var].get_function_values( this->transp_solution_local , concentration);
            fe_values_local[vector_var].get_function_values( this->darcy_solution_local , velocities);


            for (unsigned int q=0; q<n_q_points; ++q) {
                if(Transport) {
                    dispersion_tensor = get_local_dispersion_tensor(velocities[q]);
                    double Pe_local = ( velocities[q].norm() ) / ( linfty_norm(dispersion_tensor) );
                    if(Pe_local > Pe_global){
                        Pe_global = Pe_local * 3.67;
                    }
                }
                else {
                    viscosity = viscosity_res * pow( concentration[q] * ( pow(viscosity_ratio, 1./4. ) - 1.) + 1., -4.);
                }
                const double JxW = fe_values_local.JxW(q);
                for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                    const Tensor<1, dim> phi_i_s = fe_values_local[vector_var].value(i, q);
                    const double div_phi_i_s = fe_values_local[vector_var].divergence(i, q);
                    const double phi_i_c = fe_values_local[scalar_var].value(i, q);
                    const Tensor<1, dim> grad_phi_i_c = fe_values_local[scalar_var].gradient(i, q);
                    auto curl_phi_i_s = fe_values_local[vector_var].curl(i, q);

                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                        const Tensor<1, dim> phi_j_s = fe_values_local[vector_var].value(j, q);
                        const double div_phi_j_s = fe_values_local[vector_var].divergence(j, q);
                        const double phi_j_c = fe_values_local[scalar_var].value(j, q);
                        const Tensor<1, dim> grad_phi_j_c = fe_values_local[scalar_var].gradient(j, q);
                        auto curl_phi_j_s = fe_values_local[vector_var].curl(j, q);

                        if(Transport){
                            A_matrix(i, j) += (dt * ( invert(dispersion_tensor) *   phi_j_s *      phi_i_s
                                                       -                            phi_j_c *  div_phi_i_s
                                                       -                        div_phi_j_s *      phi_i_c
                                                       + velocities[q] *            phi_j_c * grad_phi_i_c // Egger
                            )               - fi * phi_j_c *      phi_i_c
                                              ) * JxW;

                            A_matrix(i, j) += delta1*dt*dispersion_tensor* (invert(dispersion_tensor)* phi_j_s + grad_phi_j_c)
                                                                         * (invert(dispersion_tensor)* phi_i_s + grad_phi_i_c) * JxW;
                        } else {
                            A_matrix(i, j) +=       ( (viscosity/k_perm) * phi_j_s *     phi_i_s
                                                     -                      phi_j_c * div_phi_i_s
                                                     -                  div_phi_j_s *     phi_i_c
                                                     ) * JxW;

                            A_matrix(i, j) += delta1 * (k_perm/viscosity)  * ((viscosity/k_perm) * phi_j_s + grad_phi_j_c)
                                                                                  * ((viscosity/k_perm) * phi_i_s + grad_phi_i_c) * JxW;

                            A_matrix(i,j) += delta2 * (k_perm/viscosity)  * div_phi_j_s * div_phi_i_s * JxW;  // 0.5 * (div u, div v)

                            A_matrix(i,j) += delta3 * (k_perm/viscosity)  * curl_phi_i_s * curl_phi_j_s * JxW;  // 0.5 * (||K|| rot Au, rot Av)
                        }
                    }
                    if(Transport){
                        F_vector(i) -= fi * concentration[q] * phi_i_c  * JxW;
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
                                if (face_vels[q] * normal >= 0)   A_matrix(i, j)  -=  face_vels[q] * normal * phi_j_c * phi_i_c * dt * JxW; // Egger
                            }
                        }
                    }
                }
            }
            double tr;  /// Reduce the conditionals in loops
            if(Transport) {
                tr = 1;
            } else {
                tr = 0;
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


                                    if(face_vels[q] * normal < 0)   B_matrix(i, j)  -= tr * face_vels[q] * normal * phi_j_m * phi_i_c * JxW; // Egger
                                    else                            BT_matrix(j, i) += tr * face_vels[q] * normal * phi_j_m * phi_i_c * JxW; // Egger

                            }
                        }
                        if(Transport) {
                            for (unsigned int i = 0; i < dofs_multiplier; ++i) {
                                const double phi_i_m = fe_face_values.shape_value(i, q);
                                for (unsigned int j = 0; j < dofs_multiplier; ++j) {
                                    const double phi_j_m = fe_face_values.shape_value(j, q);
                                    if(face_vels[q] * normal < 0)     C_matrix(i, j) -= face_vels[q] * normal * phi_j_m * phi_i_m  * JxW; // Egger
                                }
                            }
                        }
//                        else {
//                            if (cell->face(face_n)->at_boundary()) {
//                                if ( face_n == 0 ) {          /// Neumann condition (IN)  -  SLAB
//                                    Tensor<1, dim> neumann_cond;
//                                    for ( int i = 0; i < dim; i++ )
//                                        neumann_cond[i] = 1. ;
//                                    for ( unsigned int i = 0; i < dofs_multiplier; ++i ) {
//                                        const double phi_i_m = fe_face_values.shape_value(i, q);
//                                        G_vector(i) += phi_i_m * (neumann_cond * normal) * JxW;
//                                    }
//                                }
//                            }
//                        }
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
        cout << "Peclet: " << Pe_global << endl;
    }
};

int main () {

    const int dim = 2, degree = 1;
    const double final_time = 1.4;

    UnifiedProblem<dim> unified(degree);
    unified.run(  0.0, 1.0, final_time );

    return 0;
}
