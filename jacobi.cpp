#include <iostream>
#include <armadillo>
#include <math.h>
#include <stdlib.h>
using namespace std;
using namespace arma;

/* This program numerically solves Schrodinger's equation as constructed in a linear algebra formulation
 * that allows us to solve via Jacobi's method using n step points*/

int main()
{
    //i,j indexes the matrix elements we are currently performing the algorithm on (in the for and while loops) and n is the size of our N x N matrix, A. rho_min and rho_max define our step length h. p,q will determine the rotatation element for each iteration of the Jacobi algorithm. Matrix A represents the Schrodinger matrix and matrix eigen represents the matrix containing the eigenstates that will be important for plotting the omega_r dependence in the later sections of the assignment.
    int p;
    int q;
    double rho_max;
    double rho_min;
    
    cout << "Enter value for n:" << endl;
    cin >> n;
    
    mat A = zeros<mat>(n,n);
    //h is used as the increment counter between approximation terms and defined as 1/(n+1)
    double h;
    
    //harmonic oscillator potential V
    double pot_e=0;
    
    //terms used to calculate the tridiagonal values for matrix A. Also initialize the terms for the potential
    double ei;
    double dh;
    double di;
    double rho;
    double w_r;
    
    //set rho min/max values and w_r
    rho_min = 0.0;
    rho_max = 5.0;
    w_r = 5.0;
    
    //define matrix A constant terms
    h = (rho_max - rho_min) / (n+1);
    ei = -1.0/(h*h);
    dh = 2.0/(h*h);
    
    //initialize matrix A
    for(int i=0; i < n; i++){
        rho = rho_min + (i+1)*h;
        pot_e = (w_r*w_r)*(rho*rho) + 1/rho;
        //  pot_e = rho*rho;
        for(int j=0; j < n; j++){
            if(i == j){
                di = dh + pot_e;
                A(i,j) = di;
            }
            else if(fabs(i-j) == 1){
                A(i,j) = ei;
            }
        }
    }
    wall_clock armasolve;
    armasolve.tic();
    
    //use armadillo to solve for the eigenvectors of A in ascending order of eigenvalues and then print out the square of the eigenvectors to plot the probability distribution
    vec aeig = vec(n);
    mat eigvec = mat(n,n);
    eig_sym(aeig, eigvec,A);
    armasolve.toc();
    double armasolvetime =  armasolve.toc();
    
    ofstream myfile;
    myfile.open ("w=5probdist.txt");
    myfile << square(eigvec) << endl;
    myfile.close();
    
    //threshold for completion of algorithm. when largest offidag element less than epsilon, diagonalization is finished.
    double epsilon;
    epsilon = 1e-8;
    
    //eigenvalue matrix
    mat eigen = randu<mat>(n,n);
    
    //setup of eigenvector matrix, beginning as identity matrix
    for(int i=0; i < n; i++){
        for (int j = 0; j < n; j++){
            if(i==j){
                eigen(i,j) = 1.0;
            }
            else{
                eigen(i,j)= 0.0;
            }
        }
    }
    
    //set up timer for Jacobi algorithm
    wall_clock jacobisolve;
    jacobisolve.tic();
    
    //stops function from running continuously
    int iterations = 0;
    double max_iterations = (double) n * (double) n * (double) n;
    
    //variables to determine the max a_ij off diagonal element for given rotational iteration.
    double maxelement = 999; //= max(A,&p,&q,n);
    double tempmax = 0.0;
    p= 0;
    q = 1;
    //initialize sequence for rotating around largest a_ij until all offdiagonal elements are below epsilon
    while (fabs(maxelement) > epsilon && (double)iterations < max_iterations){
        
        
        //sin and cos defined as well as their relations, t and tau
        double s,c;
        double t, tau;
        
        // rotate A, eigen to get off diag elements below threshold, effectively zero;
        if (A(p,q) !=0.0){
            tau = (A(q,q) - A(p,p))/(2.0*A(p,q));
            if (tau > 0){
                t = 1.0/(tau + sqrt((1.0 + tau*tau)));
            }
            else {
                t = -1.0/(-tau + sqrt((1.0 + tau*tau)));
            }
            c = 1/sqrt((1+t*t));
            s = t*c;
        }
        else{
            c = 1.0;
            s = 0.0;
        }
        
        //rotation variables defined
        double a_pp, a_qq, a_ip, a_iq, eigen_ip, eigen_iq,a_pq;
        a_pp = A(p,p);
        a_qq = A(q,q);
        a_pq = A(p,q);
        
        //rotation and writing the pivots to the actual matrix A
        A(p,p) = c*c*a_pp - 2.0*c*s*A(p,q) + s*s*a_qq;
        A(q,q) = s*s*a_pp + 2.0*c*s*A(p,q) + c*c*a_qq;
        A(p,q) = 0.0;
        A(q,p) = 0.0;
        for (int i =0; i < n; i++){
            if (i != p && i != q){
                a_ip = A(i,p);
                a_iq = A(i,q);
                A(i,p) = c*a_ip - s*a_iq;
                A(p,i) = A(i,p);
                A(i,q) = c*a_iq + s*a_ip;
                A(q,i) = A(i,q);
            }
            
            //eigenvectors are now computed
            eigen_ip = eigen(i,p);
            eigen_iq = eigen(i,q);
            eigen(i,p) = c*eigen_ip - s*eigen_iq;
            eigen(i,q) = c*eigen_iq + s*eigen_ip;
        }
        tempmax = 0.0;
        for (int i=0; i < n; i++){
            for(int j = i + 1; j < n;j++){
                if ( (fabs(A(i,j))) > tempmax ){
                    tempmax = fabs(A(i,j));
                    p = i;
                    q = j;
                }
            }
        }
        maxelement = tempmax;
        iterations++;
    }
    
    vec eig = diagvec(A);
    vec eig_order = sort(eig);
    
    //loop over eigenvalue vector to get lowest 3 eigenvalue positions in eigen.
    int min0;
    int min1;
    int min2;
    for (int i=0; i < n; i++){
        if ( eig(i)==eig_order(0)){
            min0 = i;
        }else if(eig(i)==eig_order(1)){
            min1 = i;
        }else if(eig(i) == eig_order(2)){
            min2 = i;
        }
    }
    
    //assign lowest eigenvectors in matrix to own vectors
    vec u_0 = eigen.col(min0);
    vec u_1 = eigen.col(min1);
    vec u_2 = eigen.col(min2);
    
    //square the matrix in order to turn wave equation into probability distribution function
    for (int i =0; i < n; i++){
        u_0(i) = u_0(i)*u_0(i);
        u_1(i) = u_1(i)*u_1(i);
        u_2(i) = u_2(i)*u_2(i);
    }
    
    //normalize probability distribution function to 1. Or should it be normalized to 2\rho_max??? from Schrodinger equation, Psi should be normalized by sqrt(2/L)
    vec u_0norm = normalise(u_0,1);
    vec u_1norm = normalise(u_1,1);
    vec u_2norm = normalise(u_2,1);
    
    //end timer for jacobi algorithm
    jacobisolve.toc();
    double jacobisolvetime = jacobisolve.toc();
    
    //save the files comparing the jacobi eigenvector method and the armadillo method (saved above). The text files are written for each of the 4 cases of w_r [.01, .5, 1, 5]
    myfile.open ("w=001eigu0-1-2.txt");
    myfile << "eigenvalue"<< "rho"<< "eigen(i,min0)" << "eigen(i,min1)" << "eigen(i,min2)" << endl;
    
    //the files contain the ordered eigenvalues first, followed by the first 3 energy states of the potential well.
    myfile << eig_order<<endl;
    myfile <<"u0" << u_0norm <<endl;
    myfile <<"u1" << u_1norm << endl;
    myfile <<"u2" << u_2norm << endl;
    myfile.close();
    
    //here we output the diagonalized matrix A and the eigenvalues. There is also a comparison between the jacobi solve time and the armadillo function solve time
    cout << A << endl;
    cout << eigen << eig <<"sorted"<< eig_order<< endl;
    cout << "number of iterations is" << " " << iterations <<endl;
    cout << "armadillo eigenvector solver took:" << armasolvetime << "seconds."<< endl;
    cout << "my jacobi solver took:" << jacobisolvetime << "seconds."<< endl;
    return 0;
}

