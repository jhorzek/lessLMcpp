#include "linear_regression.h"

double sumSquaredError(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
)
{
  // compute the sum of squared errors:
  arma::mat sse = arma::trans(y - X * b) * (y - X * b);

  // other packages, such as glmnet, scale the sse with
  // 1/(2*N), where N is the sample size. We will do that here as well

  sse *= 1.0 / (2.0 * y.n_elem);

  // note: We must return a double, but the sse is a matrix
  // To get a double, just return the single value that is in
  // this matrix:
  return (sse(0, 0));
}

arma::rowvec sumSquaredErrorGradients(
    arma::colvec b, // the parameter vector
    arma::colvec y, // the dependent variable
    arma::mat X     // the design matrix
)
{
  // note: we want to return our gradients as row-vector; therefore,
  // we have to transpose the resulting column-vector:
  arma::rowvec gradients = arma::trans(-2.0 * X.t() * y + 2.0 * X.t() * X * b);

  // other packages, such as glmnet, scale the sse with
  // 1/(2*N), where N is the sample size. We will do that here as well

  gradients *= (.5 / y.n_rows);

  return (gradients);
}

arma::mat approximateHessian(arma::colvec b, // the parameter vector
                             arma::colvec y, // the dependent variable
                             arma::mat X,    // the design matrix
                             double eps      // controls the exactness of the approximation
)
{
  int nPar = b.n_elem;
  arma::mat hessian(nPar, nPar, arma::fill::zeros);

  arma::colvec stepLeft = b,
               twoStepLeft = b,
               stepRight = b,
               twoStepRight = b;

  arma::rowvec gradientsStepLeft(nPar);
  arma::rowvec gradientsTwoStepLeft(nPar);
  arma::rowvec gradientsStepRight(nPar);
  arma::rowvec gradientsTwoStepRight(nPar);

  // THE FOLLOWING CODE IS ADAPTED FROM LAVAAN.
  // SEE lavaan:::lav_model_hessian FOR THE IMPLEMENTATION
  // BY Yves Rosseel

  for (int p = 0; p < nPar; p++)
  {

    stepLeft.at(p) -= eps;
    twoStepLeft.at(p) -= 2 * eps;
    stepRight.at(p) += eps;
    twoStepRight.at(p) += 2 * eps;

    // step left
    gradientsStepLeft = sumSquaredErrorGradients(stepLeft, y, X);

    // two step left
    gradientsTwoStepLeft = sumSquaredErrorGradients(twoStepLeft, y, X);

    // step right
    gradientsStepRight = sumSquaredErrorGradients(stepRight, y, X);

    // two step right
    gradientsTwoStepRight = sumSquaredErrorGradients(twoStepRight, y, X);

    // approximate hessian
    hessian.col(p) = arma::trans((gradientsTwoStepLeft -
                                  8.0 * gradientsStepLeft +
                                  8.0 * gradientsStepRight -
                                  gradientsTwoStepRight) /
                                 (12.0 * eps));

    // reset
    stepLeft.at(p) += eps;
    twoStepLeft.at(p) += 2 * eps;
    stepRight.at(p) -= eps;
    twoStepRight.at(p) -= 2 * eps;
  }
  // make symmetric
  hessian = (hessian + arma::trans(hessian)) / 2.0;

  return (hessian);
}

// Using glmnet
lessSEM::fitResults penalizeGlmnet(arma::colvec y,
                                   arma::mat X,
                                   lessSEM::numericVector startingValues,
                                   std::vector<std::string> penalty,
                                   arma::rowvec lambda,
                                   arma::rowvec theta,
                                   arma::mat initialHessian)
{
  // With that, we can create our model:
  linearRegressionModel linReg(y, X);

  // You should also provide an initial Hessian as this can
  // improve the optimization considerably. For our model, this
  // Hessian can be computed with the Hessian function defined
  // in linerRegressionModel.cpp. For simplicity, we will
  // use the default Hessian as an example below. This will be
  // an identity matrix which will work in our simple example but may fail
  // in other cases. In practice, you should consider using a better approach.
  // arma::colvec val = Rcpp::as<arma::colvec>(startingValues);
  // arma::mat initialHessian = approximateHessian(val,
  //                                               y,
  //                                               X,
  //                                               1e-5
  // );

  // and optimize:
  lessSEM::fitResults fitResult_ = lessSEM::fitGlmnet(
      linReg,
      startingValues,
      penalty,
      lambda,
      theta,
      initialHessian, // optional, but can be very useful
      lessSEM::controlGlmnetDefault(),
      0);

  return (fitResult_);
}

// Setting up optimization with Ista works similarly:

// Using ista
lessSEM::fitResults penalizeIsta(arma::colvec y,
                                 arma::mat X,
                                 lessSEM::numericVector startingValues,
                                 std::vector<std::string> penalty,
                                 arma::rowvec lambda,
                                 arma::rowvec theta)
{

  // With that, we can create our model:
  linearRegressionModel linReg(y, X);

  // and optimize:
  lessSEM::fitResults fitResult_ = lessSEM::fitIsta(
      linReg,
      startingValues,
      penalty,
      lambda,
      theta);

  return (fitResult_);
}