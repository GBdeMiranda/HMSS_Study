#include <fstream>
#include <iostream>
#include <cmath>
#include "DarcyBase.h"

using namespace dealii;
using namespace std;

double conductivity( bool down, bool left ){
    if( down ){
        if( left )       return 10e1;  /// down-left   #01 - bool 11
        else             return 10e2;  /// down-right  #02 - bool 10
    } else {
        if( !left )      return 10e3;  /// up-right    #03 - bool 00
        else             return 10e0;  /// up-left     #04 - bool 01
    }
}

template <int dim>
class DirichletBoundary : public Function<dim> {
public:
    DirichletBoundary () : Function<dim>(1) {}

    virtual double value (const Point<dim>   &p, const unsigned int  /*component = 0*/) const    {
        double return_value = cos(M_PI*p[0])*cos(M_PI*p[1]);

        return return_value;
    }
};

template<int dim>
class NeummanBoundary : public Function<dim> {
public: NeummanBoundary() : Function<dim>() {}

    virtual Tensor<1, dim> gradient(const Point<dim> &p, bool down, bool left ) const {
        Tensor<1, dim> return_gradient;

        for (unsigned int i = 0; i < dim; ++i) {
            return_gradient[i] = M_PI;
            for (unsigned int j = 0; j < dim; ++j) {
                if (i == j)     return_gradient[i] *= sin(M_PI * p[j]);
                else            return_gradient[i] *= cos(M_PI * p[j]);
            }
        }
        return conductivity( down, left ) * return_gradient;
    }
};

template <int dim>
class SourceTerm : public Function<dim> {
public:
    SourceTerm (const double K) : Function<dim>(1), perm(K) {}
    virtual double value (const Point<dim>   &p, const unsigned int  /*component = 0*/) const {

        return 2.0*perm*M_PI*M_PI*cos(M_PI*p[0])*cos(M_PI*p[1]);
    }
private:
    const double perm;
};


template<int dim>
class ExactSolution : public Function<dim> {
public:
    ExactSolution(  bool down, bool left ) : Function<dim>(dim + 1) , down(down), left(left) {}
    virtual void vector_value(const Point<dim> &p, Vector<double> &values) const {
        double cond = conductivity( down, left);
        for (unsigned int i = 0; i < dim; ++i) {
            values[i] = cond * M_PI;
            for (unsigned int j = 0; j < dim; ++j) {
                if (i == j)     values[i] *= sin(M_PI * p[j]);
                else            values[i] *= cos(M_PI * p[j]);
            }
        }

        values[dim] = 1.0;
        for (int i = 0; i < dim; ++i) {
            values[dim] *= cos(M_PI * p[i]);
        }
    }
    bool down, left;
};


template <int dim>
class AntoniettiHeltai : DarcyBase<dim> {
public:

    AntoniettiHeltai(const unsigned int degree) : DarcyBase<dim>(degree) {}

    void run(int refinement, ConvergenceTable &convergence_table) {
        make_grid_and_dofs(refinement);
        assemble_system(true);
        this->solve();
        assemble_system(false);
//        this->compute_errors(convergence_table ,  ExactSolution<dim>() );
        compute_errors_by_domain( convergence_table );
//        this->output_results(refinement);
    }

private:

    string filename_dir = "output/Heterogeneous/";

    bool c_down(const typename DoFHandler<dim>::cell_iterator &cell) {
        return cell->center()[1] < 0.0;
    }
    bool c_left(const typename DoFHandler<dim>::cell_iterator &cell) {
        return cell->center()[0] < 0.0;
    }

    void make_grid_and_dofs (int i)  {
        GridGenerator::hyper_cube (this->triangulation, -1.0, 1.0);
        this->triangulation.refine_global (i);

        /// Set right boundary neumann
        for ( const auto &cell : this->triangulation.cell_iterators() ){
            if (cell->face(1)->at_boundary() ) {
//                cout << " CELULA " << cell->index()  << endl;
                cell->face(1)->set_boundary_id( 1 );
            }
        }

        this->dof_handler.distribute_dofs (this->fe);
        this->dof_handler_local.distribute_dofs(this->fe_local);

        cout << "# Active Cells: \t\t\t" << this->triangulation.n_active_cells() << endl
             << "# DoF for the multiplier: \t" << this->dof_handler.n_dofs() << endl;

        this->solution.reinit (this->dof_handler.n_dofs());
        this->system_rhs.reinit (this->dof_handler.n_dofs());
        this->solution_local.reinit (this->dof_handler_local.n_dofs());

        DynamicSparsityPattern dsp(this->dof_handler.n_dofs());
        DoFTools::make_sparsity_pattern (this->dof_handler, dsp, this->constraints, false);
        this->sparsity_pattern.copy_from(dsp);
        this->system_matrix.reinit(this->sparsity_pattern);

        this->constraints.clear();
        DoFTools::make_hanging_node_constraints (this->dof_handler, this->constraints);
        typename FunctionMap<dim>::type boundary_functions;
        DirichletBoundary<dim> solution_function;
        boundary_functions[0] = &solution_function;
        VectorTools::project_boundary_values (this->dof_handler, boundary_functions, QGauss<dim-1>(this->fe.degree+1), this->constraints);
        this->constraints.close();
    }


    void assemble_system (bool globalProblem)  {

        const double del1 = -0.5;
        const double del2 =  0.5;
        const double del3 =  0.5;

        FEValues<dim>     fe_values_local(this->fe_local, this->quadrature, update_values | update_gradients | update_quadrature_points | update_JxW_values);
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

        FullMatrix<double>   aux_matrix (dofs_local_per_cell, dofs_multiplier);
        Vector<double>       aux_vector (dofs_local_per_cell);
        vector<double>       mult_values(n_face_q_points);

        vector<double>          st_values (n_q_points);
        vector<Vector<double>> boundary_values (n_face_q_points,Vector<double>(dim));
        double K;
        double A;

        const FEValuesExtractors::Vector velocities (0);
        const FEValuesExtractors::Scalar pressure (dim);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            bool c_down = cell->center()[1] < 0.0, c_left = cell->center()[0] < 0.0; // cell position in domain

            typename DoFHandler<dim>::active_cell_iterator loc_cell (&this->triangulation, cell->level(), cell->index(), &this->dof_handler_local);
            fe_values_local.reinit (loc_cell);

            K = conductivity(c_down, c_left);
            const SourceTerm<dim> source_term(K);
            source_term.value_list (fe_values_local.get_quadrature_points(), st_values);
            A = 1/(K);

            A_matrix    = 0;
            F_vector    = 0;
            B_matrix    = 0;
            BT_matrix   = 0;
            U_vector    = 0;
            C_matrix    = 0;
            G_vector    = 0;

            for (unsigned int q=0; q<n_q_points; ++q) {
                double jxw = fe_values_local.JxW(q);

                for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                    const Tensor<1,dim> phi_i_u        = fe_values_local[velocities].value (i, q);
                    const Tensor<1,(dim==2)?1:3> rot_phi_i_u  = fe_values_local[velocities].curl (i, q);
                    const double        div_phi_i_u    = fe_values_local[velocities].divergence (i, q);
                    const double        phi_i_p        = fe_values_local[pressure].value (i, q);
                    const Tensor<1,dim> grad_phi_i_p   = fe_values_local[pressure].gradient(i, q);

                    for (unsigned int j=0; j<dofs_local_per_cell; ++j) {
                        const Tensor<1,dim> phi_j_u      = fe_values_local[velocities].value (j, q);
                        const Tensor<1,(dim==2)?1:3> rot_phi_j_u  = fe_values_local[velocities].curl (j, q);
                        const double        div_phi_j_u  = fe_values_local[velocities].divergence (j, q);
                        const double        phi_j_p      = fe_values_local[pressure].value (j, q);
                        const Tensor<1,dim> grad_phi_j_p = fe_values_local[pressure].gradient(j, q);

                        A_matrix(i,j) += phi_i_u * A * phi_j_u * jxw;     // ( u, v)
                        A_matrix(i,j) -= div_phi_i_u * phi_j_p * jxw;     // -(p, div v)
                        A_matrix(i,j) -= phi_i_p * div_phi_j_u * jxw;     // -(q, div u)

                        A_matrix(i,j) += del1 * K * (A * phi_j_u + grad_phi_j_p) * (A * phi_i_u + grad_phi_i_p) * jxw;  // -0.5 * K (gradp + Au) * (gradq + Av)

                        A_matrix(i,j) += del2 * (A) * div_phi_j_u * div_phi_i_u * jxw;  // 0.5 * (div u, div v)

                        A_matrix(i,j) += del3 * (A) * rot_phi_i_u * rot_phi_j_u * jxw;  // 0.5 * (||K|| rot Au, rot Av)

                    }
                    F_vector(i) -= phi_i_p * st_values[q] * jxw;    // -(f, q)

                    F_vector(i) += del2  * (A) * div_phi_i_u * st_values[q] * jxw; // 0.5 * (f, div v)

                }
            }
            if (globalProblem) {
                for (unsigned int face_n=0; face_n<GeometryInfo<dim>::faces_per_cell; ++face_n) {
                    fe_face_values.reinit (cell, face_n);
                    fe_face_values_local.reinit (loc_cell, face_n);
                    for (unsigned int q=0; q<n_face_q_points; ++q) {
                        const double JxW = fe_face_values_local.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                            const Tensor<1,dim> phi_i_u = fe_face_values_local[velocities].value (i, q);
                            const double        phi_i_p = fe_face_values_local[pressure].value (i, q);
                            for (unsigned int j=0; j<dofs_multiplier; ++j) {
                                const double  phi_j_m = fe_face_values.shape_value(j, q);

                                B_matrix(i,j) += phi_j_m * (phi_i_u*normal)*JxW;        //  (lamb,v.n)
                            }
                        }
                        if ( cell->face(face_n)->boundary_id() == 1 && cell->face(face_n)->at_boundary() ) { /// Neumann condition
                            Tensor<1, dim> neumann_cond = -NeummanBoundary<dim>().gradient(fe_face_values_local.quadrature_point(q), c_down, c_left );
//                                cout << "Celula: " << cell->index() << " | face: " << face_n << " | neuman val: " << neumann_cond * normal << endl;
                            for (unsigned int i = 0; i < dofs_multiplier; ++i) {
                                const double phi_i_m = fe_face_values.shape_value(i, q);
                                G_vector(i) += phi_i_m * neumann_cond * normal * JxW;
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

                    for (unsigned int q=0; q<n_face_q_points; ++q) {

                        const double JxW = fe_face_values.JxW(q);
                        const Tensor<1,dim> normal = fe_face_values.normal_vector(q);

                        for (unsigned int i=0; i<dofs_local_per_cell; ++i) {
                            const Tensor<1,dim> phi_i_u  = fe_face_values_local[velocities].value (i, q);
                            const double        phi_i_p  = fe_face_values_local[pressure].value (i, q);

                            F_vector(i) -= (phi_i_u * normal) * mult_values[q] * JxW;
                        }
                    }
                }
                A_matrix.gauss_jordan();
                A_matrix.vmult(U_vector, F_vector, false);
                loc_cell->set_dof_values(U_vector, this->solution_local);
            }
        }
    }

    void compute_errors_by_domain (ConvergenceTable &convergence_table) {
        const ComponentSelectFunction<dim> p_mask (dim, dim + 1);
        const ComponentSelectFunction<dim> u_mask(make_pair(0, dim), dim + 1);

        ExactSolution<dim> sol_K1(true , true );
        ExactSolution<dim> sol_K2(true , false);
        ExactSolution<dim> sol_K3(false, false);
        ExactSolution<dim> sol_K4(false, true );

        Vector<double> errors_K1 (this->triangulation.n_active_cells());
        Vector<double> errors_K2 (this->triangulation.n_active_cells());
        Vector<double> errors_K3 (this->triangulation.n_active_cells());
        Vector<double> errors_K4 (this->triangulation.n_active_cells());
        Vector<double> cellwise_errors (this->triangulation.n_active_cells());

        QGauss<dim> quad (this->degree + 2);

        double u_l2_error, p_l2_error;

        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, sol_K1,errors_K1, quad,VectorTools::L2_norm, &u_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, sol_K2,errors_K2, quad,VectorTools::L2_norm, &u_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, sol_K3,errors_K3, quad,VectorTools::L2_norm, &u_mask);
        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, sol_K4,errors_K4, quad,VectorTools::L2_norm, &u_mask);

        for (const auto &cell : this->dof_handler.active_cell_iterators()) {
            uint i = cell->index();
            if( c_down(cell) ){
                errors_K3(i)=0.;
                errors_K4(i)=0.;
            } else {
                errors_K1(i)=0.;
                errors_K2(i)=0.;
            }
            if( c_left(cell) ){
                errors_K2(i)=0.;
                errors_K3(i)=0.;
            } else {
                errors_K1(i)=0.;
                errors_K4(i)=0.;
            }
        }

        for (int i=0; i<this->triangulation.n_active_cells(); i++)
            cellwise_errors(i) = errors_K1(i) + errors_K2(i) + errors_K3(i) + errors_K4(i);
        u_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

        VectorTools::integrate_difference (this->dof_handler_local, this->solution_local, sol_K1,cellwise_errors, quad,VectorTools::L2_norm,&p_mask);
        p_l2_error = VectorTools::compute_global_error(this->triangulation, cellwise_errors, VectorTools::L2_norm);

//        cout << "Errors: ||e_u||_L2 = " << u_l2_error << ",   ||e_p||_L2 = "  << p_l2_error << endl << endl;

        convergence_table.add_value("cells", this->triangulation.n_active_cells());
        convergence_table.add_value("L2_u", u_l2_error);
        convergence_table.add_value("L2_p", p_l2_error);

    }
};

int main () {
    const int dim = 2;
    const int degree = 3;
    ConvergenceTable convergence_table;

    for (int i = 2; i < 8; ++i) {
        cout << endl << "   ---   Refinement #" << i << "   ---   " << endl;
        AntoniettiHeltai<dim> mixed_laplace_problem(degree);
        mixed_laplace_problem.run (i, convergence_table);
    }

    show_convergence( convergence_table, dim );
    return 0;
}