#ifndef LINEARREGRESSIONMODEL_H
#define LINEARREGRESSIONMODEL_H

#define USE_R 0

#include <armadillo>
#include <include/lessOptimizers.h>

double sumSquaredError(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
);

arma::rowvec sumSquaredErrorGradients(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
);

// we could also define the analytic Hessian, but we
// are a bit lazy here. The Hessian is only required
// As a starting point for the BFGS approximations. However,
// this starting point can have a huge impact.
arma::mat approximateHessian(arma::colvec b, // the parameter vector
                             arma::colvec y, // the dependent variable
                             arma::mat X,    // the design matrix
                             double eps      // controls the exactness of the approximation
);

class linearRegressionModel : public lessSEM::model
{

public:
  // the lessSEM::model class has two methods: "fit" and "gradients".
  // Both of these methods must follow a fairly strict framework.
  // First: They must receive exactly two arguments:
  //        1) an arma::rowvec with current parameter values
  //        2) an Rcpp::StringVector with current parameter labels
  //          (NOTE: the lessSEM package currently does not make use of these labels.
  //                 This is just for future use. If you don't want to use the labels,
  //                just pass any Rcpp::StringVector you want).
  // Second:
  //        1) fit must return a double (e.g., the -2-log-likelihood)
  //        2) gradients must return an arma::rowvec with the gradients. It is
  //           important that the gradients are returned in the same order as the
  //           parameters (i.e., don't shuffle your gradients, lessSEM will assume
  //           that the first value in gradients corresponds to the derivative with
  //           respect to the first parameter passed to the function).

  double fit(arma::rowvec b, lessSEM::stringVector labels) override
  {
    // NOTE: In sumSquaredError we assumed that b was a column-vector. We
    //  have to transpose b to make things work
    return (sumSquaredError(b.t(), y, X));
  }

  arma::rowvec gradients(arma::rowvec b, lessSEM::stringVector labels) override
  {
    // NOTE: In sumSquaredErrorGradients we assumed that b was a column-vector. We
    //  have to transpose b to make things work
    return (sumSquaredErrorGradients(b.t(), y, X));
  }

  // IMPORTANT: Note that we used some arguments above which we did not pass to
  // the functions: y, and X. Without these arguments, we cannot use our
  // sumSquaredError and sumSquaredErrorGradients function! To make these accessible
  // to our functions, we have to define them:

  const arma::colvec y;
  const arma::mat X;

  // finally, we create a constructor for our class
  linearRegressionModel(arma::colvec y_, arma::mat X_) : y(y_), X(X_){};
};

// Using glmnet
lessSEM::fitResults penalizeGlmnet(arma::colvec y,
                                   arma::mat X,
                                   lessSEM::numericVector startingValues,
                                   std::vector<std::string> penalty,
                                   arma::rowvec lambda,
                                   arma::rowvec theta,
                                   arma::mat initialHessian);

lessSEM::fitResults penalizeIsta(arma::colvec y,
                                 arma::mat X,
                                 lessSEM::numericVector startingValues,
                                 std::vector<std::string> penalty,
                                 arma::rowvec lambda,
                                 arma::rowvec theta);
#endif