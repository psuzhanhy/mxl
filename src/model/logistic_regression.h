#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "param.h"
#include "matvec.h"
#include "logistic.h"
#include "common.h"
#include "opthistory.h"


class LogisticRegression : public Logistic 
{
    private:
        Beta _beta;
        std::vector<double> _intercept;
        double _l1Lambda;

    public:
        LogisticRegression(CSR_matrix xf, std::vector<int> lbl, 
            int numclass, int dim, double l1Lambda, bool zeroinit);

        void multinomialProb(int sampleID, std::vector<double> &classProb, 
            Beta &beta, std::vector<double>& intercept);

        void stochasticGradient(int sampleID, 
                Beta& beta, std::vector<double> &intercept,
                Beta& betaGrad, std::vector<double> &interceptGrad);
                
 	    void fit_by_SGD(double initStepSize, int batchSize, int maxIter, 
                OptHistory &history, bool writeHistory);  

        void fit_by_AGD(double initStepSize, int maxIter, 
                OptHistory &history, bool writeHistory);

		virtual double negativeLogLik() override;

        double negativeLogLik(Beta& beta, std::vector<double> &intercept);

        double objValue(Beta& beta, std::vector<double> &intercept);

        double l1Regularizer(Beta& beta) const;

        void proximalL1(Beta& beta);

        void subgradientL1Regularizer(Beta& beta, Beta& betaGrad);

        void setL1Lambda(double l1Lambda);

        Beta getBeta();

        std::vector<double> getIntercept();
};

#endif