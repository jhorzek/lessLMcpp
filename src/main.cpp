#include "linear_regression.h"

int main(int argc, char *argv[])
{

    if (argc != 5)
        error("Expecting three arguments: X-file, y-file, lambda, and optimizer");

    print << "X: "         << argv[1] << std::endl;
    print << "y:"          << argv[2] << std::endl;
    print << "lambda: "    << argv[3] << std::endl;
    print << "optimizer: " << argv[4] << std::endl;

    std::string X_file = argv[1];
    std::string y_file = argv[2];
    double lambda_val = std::stod(argv[3]);
    std::string optimizer = argv[4];

    arma::mat X, ymat;

    X.load(X_file);
    ymat.load(y_file);

    if (ymat.n_cols != 1)
        error("y should only have one column");

    arma::colvec y = ymat.col(0);

    // add intercept
    arma::colvec ones(X.n_rows, arma::fill::ones);
    X.insert_cols(0, ones);

    // starting values
    arma::rowvec b_vec(X.n_cols, arma::fill::zeros);
    lessSEM::numericVector b(b_vec);

    // penalty
    std::vector<std::string> penalty;

    arma::rowvec lambda(1, arma::fill::zeros);
    lambda(0) = lambda_val;
    arma::rowvec theta(1, arma::fill::zeros);

    penalty.push_back("none"); // no penalty for intercept
    for (int i = 1; i < X.n_cols; i++)
    {
        penalty.push_back("lasso"); // lasso penalty for everything else
    }

    arma::mat initialHessian = approximateHessian(b.values.t(), // the parameter vector
                                                  y,            // the dependent variable
                                                  X,            // the design matrix
                                                  .0000001      // controls the exactness of the approximation
    );

    lessSEM::fitResults fitRes;

    if (optimizer == "glmnet")
    {
        fitRes = penalizeGlmnet(y,
                                X,
                                b,
                                penalty,
                                lambda,
                                theta,
                                initialHessian);
    }
    else if (optimizer == "ista")
    {
        fitRes = penalizeIsta(y,
                              X,
                              b,
                              penalty,
                              lambda,
                              theta);
    }
    else
    {
        error("Unknown optimizer. Available are glmnet or ista.");
    }
    print << fitRes.parameterValues << std::endl;

    return 0;
}