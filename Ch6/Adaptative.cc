#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/tensor_function.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/block_vector.h>
#include <deal.II/lac/sparse_direct.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/affine_constraints.h>

#include <deal.II/grid/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_refinement.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/grid_in.h>

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_renumbering.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_dg_vector.h>
#include <deal.II/fe/fe_face.h>
#include <deal.II/fe/fe_trace.h>

#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_faces.h>
#include <deal.II/numerics/error_estimator.h>
#include <deal.II/numerics/solution_transfer.h>

#include <fstream>
#include <iostream>
#include <cmath>

using namespace dealii;
using namespace std;

const double alpha_mol      = 1.8e-8;
const double alpha_l        = 1.8e-8;
const double alpha_t        = 1.8e-9;
const double fi             = 1.0;
const double permeability   = 1.e0;
const double mures          = 0.001;
const double mu0            = 0.05;
//    const double mob            = mu0/mures;
const double L              = 1.0;

template <int dim>
void print_mesh_info(const Triangulation<dim> &triangulation, const string & name, int deg, int mesh){
    string filename = name + "_d" + Utilities::to_string(deg, 0) + "_m" + Utilities::to_string(mesh, 0) + ".eps";
    cout << "Mesh info:" << endl << " dimension: " << dim << endl
    << " no. of cells: " << triangulation.n_active_cells() << endl;
    map<types::boundary_id, unsigned int> boundary_count;
    for (auto &c : triangulation.active_cell_iterators()) {
        for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face) {
            if (c->face(face)->at_boundary())
                boundary_count[c->face(face)->boundary_id()]++;
        }
    }
    cout << " boundary indicators: ";
    for (const pair<const types::boundary_id, unsigned int> &pair : boundary_count) {
        cout << pair.first << "(" << pair.second << " times) ";
    }
    cout << endl;
    ofstream out(filename);
    GridOut grid_out;
    grid_out.write_eps(triangulation, out);
    cout << " written to " << filename << endl << endl;
}

template <int dim>
class InitialCondition : public Function<dim> {
public:
    InitialCondition () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p, const unsigned int  component = 0) const {
        if (p[0]<1.e-10)
            return 1.0;
        else
            return 0.0;
    }
};

template <int dim>
class BoundarySlabConcentration : public Function<dim> {
public:
    BoundarySlabConcentration () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p, const unsigned int  component = 0) const {
        return 1.0;
    }
};

template <int dim>
class BoundarySlabPressure_out : public Function<dim> {
public:
    BoundarySlabPressure_out () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p, const unsigned int  component = 0) const {

        return 0.0;
    }
};

template <int dim>
class BoundarySlabPressure_in : public Function<dim>{
public:
    BoundarySlabPressure_in () : Function<dim>(1) {}

    virtual double value (const Point<dim> &p, const unsigned int  component = 0) const {
        return 0.05*mu0*L;
    }
};

template <int dim>
class MixedLaplaceProblem{
public:
    MixedLaplaceProblem (const unsigned int degree)
        :
        degree (degree),
        quadrature_formula(degree+2),
        face_quadrature_formula(degree+2),
        fe_local (FE_DGRaviartThomas<dim>(degree), 1,
                  FE_DGQ <dim>(degree), 1),
        dof_handler_local (triangulation),
        fe (degree),
        dof_handler (triangulation)
    {}

    void run (const double final_time )
    {

        vector< unsigned int > repetitions(dim);
        repetitions[0] = 20;
        repetitions[1] = 5;
        
        double x_init  = 0.0;
        double x_final = L;
        double y_init  = 0.0;
        double y_final = 0.25;
        
        Point<dim> p1(x_init, y_init);
        Point<dim> p2(x_final, y_final);
        
        
        GridGenerator::subdivided_hyper_rectangle(triangulation, repetitions,p1, p2, false);
        //        GridGenerator::hyper_cube (triangulation, 0, 1.0);
        //        triangulation.refine_global (i);
        //        GridGenerator::subdivided_hyper_cube(triangulation, 40, 0.0, 1000);

        make_grid_and_dofs();

//        InitialCondition<dim> initial_condition;
//        constraints.close();
//        VectorTools::project (dof_handler_local,
//                              constraints,
//                              QGauss<dim>(degree+2),
//                              initial_condition,
//                              last_Transport_solution_local);


        double current_time  = 0.0;
        int step = 0;
        int iter = 0;

        change_boundary(false);
        
        //Darcy
        assemble_system (true, false);
        solve (false);
        assemble_system (false, false);
        
        system_rhs.reinit(dof_handler.n_dofs());
        system_matrix.reinit(sparsity_pattern);

        while( current_time < final_time ) {
            
//            if(viscosity_convergence() )
//            {
                current_time += dt/2.;

//                Transport_solution_local=last_Transport_solution_local;
            
//                for (const auto &cell : dof_handler.active_cell_iterators())
//                {
//                    vector<double> viscosities( quadrature_formula.size(),0.0);
//                    global_viscosities.at(cell->index()) = viscosities;
//                    last_global_viscosities.at(cell->index()) = viscosities;
//                }
            
            //Transport
            change_boundary(true);
            
            assemble_system (true, true);
            solve (true);
            assemble_system (false, true);
        
            //Transport -- c^{n+1} = 2c^{n+1/2} - c^{n}
            last_Transport_solution_local.add(2., Transport_solution_local, -2., last_Transport_solution_local);

            system_rhs.reinit(dof_handler.n_dofs());
            system_matrix.reinit(sparsity_pattern);
            
            current_time += dt/2.;

//            hrefinement();
            
            if((iter % 10)==0){
                //                    output_results (false,step);
                output_results (true,step);
                iter = 0;
            }
            iter++;

            //Darcy
            change_boundary(false);
            
            assemble_system (true, false);
            solve (false);
            assemble_system (false, false);
            
            system_rhs.reinit(dof_handler.n_dofs());
            system_matrix.reinit(sparsity_pattern);

            cout << "   Current time:  " << current_time << endl;
            
            //                output_results (false,step);
            step++;
        }
    }
    
private:
    const unsigned int   degree;

    Triangulation<dim>   triangulation;
    QGauss<dim>          quadrature_formula;
    QGauss<dim-1>        face_quadrature_formula;

    double h_mesh;
    double dt;
    double max_error;
    
    FESystem<dim>        fe_local;
    DoFHandler<dim>      dof_handler_local;
    Vector<double>       Transport_solution_local;
    Vector<double>       last_Transport_solution_local;
    Vector<double>       Darcy_solution_local;

    FE_FaceQ<dim>        fe;
    DoFHandler<dim>      dof_handler;
    Vector<double>       solution;
    Vector<double>       system_rhs;

    SparsityPattern      sparsity_pattern;
    SparseMatrix<double> system_matrix;

    AffineConstraints<double> constraints;

    string filename_trans = "output/Transport/";
    string filename_darcy = "output/Darcy/";
    
//    vector< vector<double> >  global_viscosities;
//    vector< vector<double> >  last_global_viscosities;
    
//    bool viscosity_convergence(){
//        double tol = 1e-10;
//        double error = 0.0;
//        for(uint ii = 0; ii < triangulation.n_active_cells(); ii++) {
//            for(uint jj = 0; jj < quadrature_formula.size() ; jj++) {
//                //                    if( fabs( global_viscosities[ii][jj] - last_global_viscosities[ii][jj] ) > error )  error = fabs( global_viscosities[ii][jj] - last_global_viscosities[ii][jj] ); /// Norma Max
//                error += pow( global_viscosities[ii][jj] - last_global_viscosities[ii][jj], 2 );  /// Norma 2
//            }
//        }
//        error = sqrt(error);
//        last_global_viscosities = global_viscosities;
//        cout << "  \t Error: " << error << endl;
//        return (error < tol);
//    }


    void make_grid_and_dofs ()    {
        
        dof_handler.distribute_dofs (fe);
        dof_handler_local.distribute_dofs(fe_local);

        h_mesh = GridTools::maximal_cell_diameter(triangulation)*sqrt(2.)/2.;
        dt = 0.01;

        for (const auto &cell : dof_handler.active_cell_iterators())         {
            if(cell->face(0)->at_boundary())
                cell->face(0)->set_all_boundary_ids(1);
            if(cell->face(1)->at_boundary())
                cell->face(1)->set_all_boundary_ids(2);
        }

        solution.reinit (dof_handler.n_dofs());
        system_rhs.reinit (dof_handler.n_dofs());
        Transport_solution_local.reinit (dof_handler_local.n_dofs());
        last_Transport_solution_local.reinit (dof_handler_local.n_dofs());
        Darcy_solution_local.reinit (dof_handler_local.n_dofs());

        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        constraints.close();

        DynamicSparsityPattern dsp(dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (dof_handler, dsp, constraints, false);
        sparsity_pattern.copy_from(dsp);
        system_matrix.reinit(sparsity_pattern);
        
        cout << "Number of active cells: " << triangulation.n_active_cells() << endl
        << "Total number of cells: " << triangulation.n_cells() << endl
        << "Number of degrees of freedom for the multiplier: " << dof_handler.n_dofs() << endl
        << "Nonzero entries: " << sparsity_pattern.n_nonzero_elements() << endl
        << "Bandwidth: " << sparsity_pattern.bandwidth() << endl << endl;
        
//        global_viscosities.resize( triangulation.n_active_cells() );
//        last_global_viscosities.resize( triangulation.n_active_cells() );
//
//        for (const auto &cell : dof_handler.active_cell_iterators())
//        {
//            vector<double> viscosities( quadrature_formula.size(),0.0);
//            global_viscosities.at(cell->index()) = viscosities;
//            last_global_viscosities.at(cell->index()) = viscosities;
//        }

    }

    void change_boundary(bool Transport){
        constraints.clear();
        DoFTools::make_hanging_node_constraints(dof_handler, constraints);
        typename FunctionMap<dim>::type boundary_functions;
        
        
        BoundarySlabPressure_out<dim> boundary_solution_pressure_out;
        BoundarySlabPressure_in<dim> boundary_solution_pressure_in;
        BoundarySlabConcentration<dim> boundary_solution_concentration;

        if(Transport){
            boundary_functions[1] = &boundary_solution_concentration;
            VectorTools::project_boundary_values(dof_handler, boundary_functions, QGauss < dim - 1 > (fe.degree + 1), constraints);
        } else{
            boundary_functions[1] = &boundary_solution_pressure_in;
            VectorTools::project_boundary_values(dof_handler, boundary_functions, QGauss < dim - 1 > (fe.degree + 1), constraints);

            boundary_functions[2] = &boundary_solution_pressure_out;
            VectorTools::project_boundary_values(dof_handler, boundary_functions, QGauss < dim - 1 > (fe.degree + 1), constraints);
        }
            
        constraints.close();
    }

    
    
    
    void hrefinement () {
        QGauss<dim>   quadrature_formula(degree+2);
        QGauss<dim-1> face_quadrature_formula(degree+2);
        
        FEValues<dim> fe_values_local (fe_local, quadrature_formula,
                                       update_values    | update_gradients |
                                       update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_local (fe_local, face_quadrature_formula,
                                                update_values    | update_normal_vectors |
                                                update_quadrature_points  | update_JxW_values);
        
        
        const FEValuesExtractors::Vector sigma (0);
        const FEValuesExtractors::Scalar saturation (dim);

        Vector<float> estimated_error_per_cell(triangulation.n_active_cells());
//        KellyErrorEstimator<dim>::estimate(
//                                           dof_handler_local,
//                                           face_quadrature_formula,
//                                           map<types::boundary_id, const Function<dim> *>(),
//                                           last_Transport_solution_local,
//                                           estimated_error_per_cell,
//                                           fe_local.component_mask(saturation));
        
        
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();

        vector<Tensor<1,dim> > velocities( n_q_points);
        vector<Tensor<1,dim> > concentration_gradient( n_q_points);
        vector<Tensor<1,dim> > sigma_values ( n_q_points);
        vector<double> last_concentration( n_q_points);
        vector<double> face_concentration( n_face_q_points);
        vector<double> mult_values( n_face_q_points);

        
        int i=0;
        for (const auto &cell : dof_handler.active_cell_iterators())        {

            typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);
            fe_values_local.reinit (loc_cell);

            fe_values_local[saturation].get_function_values(last_Transport_solution_local, last_concentration);
            fe_values_local[saturation].get_function_gradients(last_Transport_solution_local, concentration_gradient);
            fe_values_local[sigma].get_function_values( last_Transport_solution_local , sigma_values);
            fe_values_local[sigma].get_function_values( Darcy_solution_local , velocities);

            double cell_error_1 = 0.0;
            double cell_error_3 = 0.0;

            for (unsigned int q=0; q<n_q_points; ++q)
            {
                Tensor<2, dim> dispersion_tensor;

                    for( int i = 0; i < dim; i++)
                        dispersion_tensor[i][i] = alpha_mol;

                    for( int ii = 0; ii < dim; ii++)
                        for( int jj = 0; jj < dim; jj++)
                            dispersion_tensor[ii][jj] += velocities[q].norm() * (
                                                                                 alpha_l * velocities[q][ii]*velocities[q][jj]/velocities[q].norm_square()
                                                                                 +  alpha_t * ( (1-fabs(ii-jj))-velocities[q][ii]*velocities[q][jj]/velocities[q].norm_square() ) );

//                } else {
//
//                    viscosity[q] = pow(pow(mures,-1./4.)*last_concentration[q]+(1-last_concentration[q])*pow(mu0,-1./4.),-4);
//                    //                    viscosity[q] = mures*pow((1.-last_solution[q]+pow(mob,1./4.)*prev_solution[q]),-4.);
//                    //                    cout << " viscosity:  "<< viscosity[q] << endl;
//                }
                cell_error_1 += (sigma_values[q] + dispersion_tensor*concentration_gradient[q])*
                                (sigma_values[q] + dispersion_tensor*concentration_gradient[q])*
                                  fe_values_local.JxW(q);
            }


                        for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n)
                        {
                            fe_face_values.reinit(cell, face_n);
                            fe_face_values_local.reinit(loc_cell, face_n);

                            fe_face_values.get_function_values (solution, mult_values);
                            fe_face_values_local[saturation].get_function_values(last_Transport_solution_local, face_concentration);
                            for (unsigned int q=0; q<n_face_q_points; ++q)
                            {

                                cell_error_3 += h_mesh*(face_concentration[q] - mult_values[q])*
                                                h_mesh*(face_concentration[q] - mult_values[q])*fe_face_values.JxW(q);
                            }
                        }

                            estimated_error_per_cell(i) = cell_error_3;
//                             cout <<estimated_error_per_cell(i)<<endl;
                            ++i;
        //
    }
        
        
//                GridRefinement::refine(triangulation,
//                                       estimated_error_per_cell,
//                                       1.e-5);
//
//                GridRefinement::coarsen(triangulation,
//                                        estimated_error_per_cell,
//                                        1.e-6);
        
//        GridRefinement::refine_and_coarsen_fixed_number(triangulation,
//                                                        estimated_error_per_cell,
//                                                        0.3,
//                                                        0.03);
        GridRefinement::refine_and_coarsen_fixed_number( triangulation, estimated_error_per_cell, 0.1, 0.09);
        //        GridRefinement::refine_and_coarsen_fixed_fraction(triangulation,
        //                                                          estimated_error_per_cell,
        //                                                          0.6,
        //                                                          0.2);
        
        if (triangulation.n_levels() > 4)
            for (const auto &cell :
                 triangulation.active_cell_iterators_on_level(4))
                cell->clear_refine_flag();
//        for (const auto &cell :
//             triangulation.active_cell_iterators_on_level(1))
//            cell->clear_coarsen_flag();
        //        i=0;
        //        for (const auto &cell : dof_handler.active_cell_iterators())
        //        {
        //            typename hp::DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);
        //            hp_fe_values_local.reinit(loc_cell);
        //            const FEValues<dim> &fe_values_local = hp_fe_values_local.get_present_fe_values();
        //
        //            if ( estimated_error_per_cell(i) > 0.5 )
        //                loc_cell->set_active_fe_index(loc_cell->active_fe_index() + 1);
        //
        ////            cout <<estimated_error_per_cell(i)<<endl;
        //
        //            ++i;
        //            cell->set_active_fe_index(cell->active_fe_index() + 1);
        //        }
        
        max_error = estimated_error_per_cell.l2_norm();
        
        triangulation.prepare_coarsening_and_refinement();
        SolutionTransfer<dim, Vector<double>> solution_transfer(dof_handler_local);
        solution_transfer.prepare_for_coarsening_and_refinement(last_Transport_solution_local);
        
        
        triangulation.execute_coarsening_and_refinement();
        
        
        dof_handler_local.distribute_dofs(fe_local);
        
        Vector<double> tmp(dof_handler_local.n_dofs());
        solution_transfer.interpolate(last_Transport_solution_local, tmp);
        
        make_grid_and_dofs();
        
        last_Transport_solution_local = tmp;
        
    }

    
    void assemble_system (bool globalProblem, bool Transport)    {
        FEValues<dim> fe_values_local (fe_local, quadrature_formula, update_values    | update_gradients |
                                 update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values (fe, face_quadrature_formula,
                                          update_values    | update_normal_vectors |
                                          update_quadrature_points  | update_JxW_values);
        FEFaceValues<dim> fe_face_values_local (fe_local, face_quadrature_formula,
                                                update_values    | update_normal_vectors |
                                                update_quadrature_points  | update_JxW_values);

        const unsigned int   dofs_local_per_cell   = fe_local.dofs_per_cell;
        const unsigned int   n_q_points      = quadrature_formula.size();
        const unsigned int   n_face_q_points = face_quadrature_formula.size();

        vector<types::global_dof_index> dof_indices (fe.dofs_per_cell);

        FullMatrix<double>   cell_matrix (fe.dofs_per_cell, fe.dofs_per_cell);
        FullMatrix<double>   A_matrix (dofs_local_per_cell, dofs_local_per_cell);
        FullMatrix<double>   B_matrix (dofs_local_per_cell, fe.dofs_per_cell);
        FullMatrix<double>   BT_matrix (fe.dofs_per_cell, dofs_local_per_cell);
        Vector<double>       F_vector (dofs_local_per_cell);
        Vector<double>       U_vector (dofs_local_per_cell);
        Vector<double>       cell_vector (fe.dofs_per_cell);

        FullMatrix<double>   aux_matrix (dofs_local_per_cell, fe.dofs_per_cell);
        Vector<double>       aux_vector (dofs_local_per_cell);

        vector<double>  mult_values(face_quadrature_formula.size());
        vector<double>  viscosity ( n_q_points );

        vector<Tensor<1,dim> > velocities( n_q_points);
        vector<Tensor<1,dim> > face_velocities( n_face_q_points);
        
        vector<double> prev_solution( n_q_points);
        vector<double> last_concentration( n_q_points);
        vector<double> face_prev_solution( n_face_q_points);
        
        const FEValuesExtractors::Vector sigma (0);
        const FEValuesExtractors::Scalar saturation (dim);

        for (const auto &cell : dof_handler.active_cell_iterators()) {

            typename DoFHandler<dim>::active_cell_iterator loc_cell (&triangulation, cell->level(), cell->index(), &dof_handler_local);
            fe_values_local.reinit (loc_cell);

            A_matrix    = 0;
            F_vector    = 0;
            B_matrix    = 0;
            BT_matrix   = 0;
            U_vector    = 0;
            cell_matrix = 0;
            cell_vector = 0;
            
            fe_values_local[saturation].get_function_values(last_Transport_solution_local, last_concentration);
//            fe_values_local[saturation].get_function_values( Transport_solution_local , prev_solution);
            fe_values_local[sigma].get_function_values( Darcy_solution_local , velocities);

            
            for (unsigned int q=0; q<n_q_points; ++q) {
                Tensor<2, dim> dispersion_tensor;
                
                if(Transport){
                    
                    for( int i = 0; i < dim; i++)
                        dispersion_tensor[i][i] = alpha_mol;
                    
                    for( int ii = 0; ii < dim; ii++)
                        for( int jj = 0; jj < dim; jj++)
                            dispersion_tensor[ii][jj] += velocities[q].norm() * (
                                                      alpha_l * velocities[q][ii]*velocities[q][jj]/velocities[q].norm_square()
                                                   +  alpha_t * ( (1-fabs(ii-jj))-velocities[q][ii]*velocities[q][jj]/velocities[q].norm_square() ) );
                    
                } else {
                    viscosity[q] = pow(pow(mures,-1./4.)*last_concentration[q]+(1-last_concentration[q])*pow(mu0,-1./4.),-4);
//                    viscosity[q] = mures*pow((1.-last_solution[q]+pow(mob,1./4.)*prev_solution[q]),-4.);
//                    cout << " viscosity:  "<< viscosity[q] << endl;
                }

                const double JxW = fe_values_local.JxW(q);
                for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                    const Tensor<1, dim> phi_i_s = fe_values_local[sigma].value(i, q);
                    const double div_phi_i_s = fe_values_local[sigma].divergence(i, q);
                    const double phi_i_c = fe_values_local[saturation].value(i, q);
                    const Tensor<1, dim> grad_phi_i_c = fe_values_local[saturation].gradient(i, q);
                            
                    for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                        const Tensor<1, dim> phi_j_s = fe_values_local[sigma].value(j, q);
                        const double div_phi_j_s = fe_values_local[sigma].divergence(j, q);
                        const double phi_j_c = fe_values_local[saturation].value(j, q);

                        if(Transport){
                            A_matrix(i, j) += 0.5*dt *  (  invert(dispersion_tensor) * phi_j_s * phi_i_s
                                                 -   phi_j_c * div_phi_i_s
                                                 -   div_phi_j_s * phi_i_c
                                                 +   velocities[q] *  phi_j_c * grad_phi_i_c
                                                     ) * JxW
                                                 -   fi * phi_j_c * phi_i_c * JxW;
                        }else{
                            A_matrix(i, j) +=       ( (viscosity[q]/permeability) * phi_j_s * phi_i_s
                                                     -   phi_j_c * div_phi_i_s
                                                     -   div_phi_j_s * phi_i_c
                                                     ) * JxW;

                         }
                    }
                        if(Transport)
                                F_vector(i) -= (fi * last_concentration[q]) * phi_i_c  * JxW;
                            
//                        }else{
//                            if (cell->index() == 0)
//                                F_vector(i) -= phi_i_c * (50. / pow(h_mesh, 2)) * JxW;
//
//                            auto final_cell = triangulation.last()->index();
//                            if (cell->index() == final_cell)
//                                F_vector(i) -= phi_i_c * (-50. / pow(h_mesh, 2)) * JxW;
//                        }
                }
            }

            if(Transport)
            {
                /// Loop nas faces dos elementos
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values_local[sigma].get_function_values( Darcy_solution_local , face_velocities);

                    for (unsigned int q=0; q<n_face_q_points; ++q) {
                        
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1, dim> normal = fe_face_values_local.normal_vector(q);
                        
                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i){
                            const double phi_i_c = fe_face_values_local[saturation].value(i, q);
                            
                            for (unsigned int j = 0; j < dofs_local_per_cell; ++j) {
                                const double phi_j_c = fe_face_values_local[saturation].value(j, q);
                                if (face_velocities[q] * normal >= 0)
                                    A_matrix(i, j)  -=  face_velocities[q] * normal * phi_j_c * phi_i_c * 0.5*dt * JxW;
                            }
                        }
                    }
                }
            }
            if (globalProblem) {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values_local[saturation].get_function_values( last_Transport_solution_local , face_prev_solution);
                    fe_face_values_local[sigma].get_function_values( Darcy_solution_local , face_velocities);

                    for (unsigned int q=0; q<n_face_q_points; ++q) {

                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_s = fe_face_values_local[sigma].value(i, q);
                            const double phi_i_c = fe_face_values_local[saturation].value(i, q);
                            
                            for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
                                const double phi_j_m = fe_face_values.shape_value(j, q);
                                
                                if(Transport) {
                                    B_matrix(i, j) +=   phi_i_s * normal * phi_j_m * 0.5*dt * JxW;
                                    if(face_velocities[q] * normal < 0)
                                        B_matrix(i, j) -= face_velocities[q] * normal * phi_j_m * phi_i_c * 0.5*dt * JxW;
                                    
                                    BT_matrix(j, i) += phi_i_s * normal * phi_j_m * 0.5*dt  * JxW;
                                    
                                    if(face_velocities[q] * normal >= 0)
                                        BT_matrix(j, i) += face_velocities[q] * normal * phi_j_m * phi_i_c * 0.5*dt * JxW;
                                
                                } else {
                                    B_matrix (i, j) +=  phi_i_s * normal * phi_j_m  * JxW;
                                    BT_matrix(j, i) +=  phi_i_s * normal * phi_j_m  * JxW;

                                }
                            }
                        }
                        if(Transport) {
                            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
                                const double phi_i_m = fe_face_values.shape_value(i, q);
                                if(cell->face(face_n)->boundary_id() == 2 && face_velocities[q] * normal >= 0)
                                    cell_vector(i) -= face_velocities[q] * normal * phi_i_m * face_prev_solution[q] * 0.5*dt * JxW;
                                
                                for (unsigned int j = 0; j < fe.dofs_per_cell; ++j) {
                                    const double phi_j_m = fe_face_values.shape_value(j, q);
                                    if(face_velocities[q] * normal < 0)
                                        cell_matrix(i, j) -= face_velocities[q] * normal * phi_j_m * phi_i_m  * 0.5*dt * JxW;
                                }
                            }
//                        } else {
//                            for (unsigned int i = 0; i < fe.dofs_per_cell; ++i) {
//                                const double phi_i_m = fe_face_values.shape_value(i, q);
//                                Tensor<1, dim> neumann_bc;
//                                neumann_bc[0]=0.05/mu0;
//                                neumann_bc[1]=0.05/mu0;
//                                if(cell->face(face_n)->boundary_id() == 1){
//                                       cell_vector(i) -= neumann_bc * normal * phi_i_m * JxW;
//                                }
//                            }
                        }
                    }
                }
            }
            else {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    fe_face_values.get_function_values (solution, mult_values);
                    fe_face_values_local[saturation].get_function_values( last_Transport_solution_local , face_prev_solution);
                    fe_face_values_local[sigma].get_function_values( Darcy_solution_local , face_velocities);

                    for (unsigned int q=0; q<n_face_q_points; ++q) {
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i = 0; i < dofs_local_per_cell; ++i) {
                            const Tensor<1, dim> phi_i_s = fe_face_values_local[sigma].value(i, q);
                            const double phi_i_c = fe_face_values_local[saturation].value(i, q);
                            
                            if(Transport) {
                                F_vector(i) -= phi_i_s * normal * 0.5*dt * mult_values[q] * JxW;
                                if(face_velocities[q] * normal < 0)
                                    F_vector(i) += face_velocities[q] * normal * phi_i_c * 0.5*dt * mult_values[q] * JxW;
                            } else {
                                F_vector(i) -= phi_i_s * normal * mult_values[q] * JxW;
                            }
                        }
                    }
                }
            }

            A_matrix.gauss_jordan();

            if (globalProblem) {
                A_matrix.vmult(aux_vector, F_vector, false);            //  A^{-1} * F
                BT_matrix.vmult(cell_vector, aux_vector, true);        //  B.T * A^{-1} * F
                A_matrix.mmult(aux_matrix, B_matrix, false);            //  A^{-1} * B
                BT_matrix.mmult(cell_matrix, aux_matrix, true);         // -C + B.T * A^{-1} * B

                cell->get_dof_indices(dof_indices);

                constraints.distribute_local_to_global (cell_matrix, cell_vector, dof_indices,system_matrix, system_rhs);
            }
            else {
                A_matrix.vmult(U_vector, F_vector, false);
                
                if(Transport) {
                    loc_cell->set_dof_values(U_vector, Transport_solution_local);
                } else {
                    loc_cell->set_dof_values(U_vector, Darcy_solution_local);
                    
//                    global_viscosities.at( cell->index() ) = viscosity;
                }
            }
        }
    }


    void solve (bool Transport)    {
        SparseDirectUMFPACK A_direct;
        A_direct.initialize(system_matrix);
        A_direct.vmult(solution, system_rhs);
        constraints.distribute(solution);
    }


    void output_results (bool Transport, int step) const    {
        if(Transport) {
            string filename = filename_trans + "Transport_step_" + dealii::Utilities::to_string(step, 0) + ".vtu";

            ofstream output (filename.c_str());

            DataOut<dim> data_out;
            vector<string> names (dim, "sigma");
            names.emplace_back("saturation");
            vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation
                (dim+1, DataComponentInterpretation::component_is_part_of_vector);
            component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
            data_out.add_data_vector (dof_handler_local, last_Transport_solution_local, names, component_interpretation);

            data_out.build_patches (fe_local.degree);
            data_out.write_vtu (output);
        }else{
            string filename = filename_darcy + "Darcy_step_" + dealii::Utilities::to_string(step, 0) + ".vtu" ;

            ofstream output (filename.c_str());
            
            DataOut<dim> data_out;
            vector<string> names (dim, "velocities");
            names.emplace_back("pressure");
            vector<DataComponentInterpretation::DataComponentInterpretation> component_interpretation
            (dim+1, DataComponentInterpretation::component_is_part_of_vector);
            component_interpretation[dim] = DataComponentInterpretation::component_is_scalar;
            data_out.add_data_vector (dof_handler_local, Darcy_solution_local, names, component_interpretation);
            
            data_out.build_patches (fe_local.degree);
            data_out.write_vtu (output);

        }

    }
    
};

int main () {
    using namespace dealii;

    const int dim = 2;

    const double final_time = 4.0;
    
        MixedLaplaceProblem<dim> mixed_laplace_problem(1);
        mixed_laplace_problem.run (final_time);

    return 0;
}
