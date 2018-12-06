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
        double _l2Lambda;

    public:
        LogisticRegression(CSR_matrix xf, std::vector<int> lbl, 
            int numclass, int dim, double l1Lambda, double l2Lambda, bool zeroinit);

        void multinomialProb(int sampleID, std::vector<double> &classProb, 
            Beta &beta, std::vector<double>& intercept);

        void stochasticGradient(int sampleID, 
                Beta& beta, std::vector<double> &intercept,
                Beta& betaGrad, std::vector<double> &interceptGrad);
                
 	    void proximalSGD(double initStepSize, std::string stepsizeRule, int batchSize,
                int maxIter, OptHistory &history, bool writeHistory, bool adaptiveStop);  

        void proximalAGD(double stepSize, int maxIter, 
                OptHistory &history, bool writeHistory);

        void proximalHybridBatchingGD(double stepSize, int maxIter, 
                OptHistory &history, bool writeHistory);

        void hybridFirstOrder(double initSGDStepSize, int batchSizeSGD, std::string stepsizeRuleSGD, double stepSizeAGD,
                int maxIter, OptHistory &history);
            
		virtual double negativeLogLik() override;

        double negativeLogLik(Beta& beta, std::vector<double> &intercept);

        double objValue(Beta& beta, std::vector<double> &intercept);

        double l1Regularizer(Beta& beta) const;

        double l2Regularizer(Beta& beta) const;

        void proximalL1(Beta& beta);

        void l1Regularizer_Subgradient(Beta& beta, Beta& betaGrad) const;

        void l2Regularizer_Gradient(Beta& beta, Beta& betaGrad) const;

        void setL1Lambda(double l1Lambda);

        Beta getBeta();

        std::vector<double> getIntercept();
};

#endif