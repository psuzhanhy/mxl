#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <sys/time.h>
#include "matvec.h"
#include "readinput.h"

CSR_matrix Dense2CSR (DenseData denseInput) 
{ 
	CSR_matrix csrX;	     	
    int nnz = 0, row_nnz = 0;
    csrX.row_offset.push_back(0);
    for(int j=0 ; j<denseInput.featureMat[0].size() ; j++) 
	{
        if( denseInput.featureMat[0][j] !=0 ){
            csrX.col.push_back(j);
            csrX.val.push_back(denseInput.featureMat[0][j]) ;
            row_nnz++;
            nnz++;
        }
    }
    for(int i=1 ; i<denseInput.featureMat.size() ; i++) 
	{
        csrX.row_offset.push_back(csrX.row_offset[i-1] + row_nnz) ;
        row_nnz = 0;
        for(int j=0 ; j< denseInput.featureMat[i].size(); j++) {
            if (denseInput.featureMat[i][j] != 0) {
                csrX.col.push_back(j);
                csrX.val.push_back(denseInput.featureMat[i][j]);
                row_nnz++;
                nnz++;                
            }    
        }
    }   
    csrX.row_offset.push_back(csrX.row_offset[denseInput.featureMat.size()-1] + row_nnz); 

    csrX.row_array_length = denseInput.featureMat.size()+1 ;  // equals to number of rows plus 1
    csrX.number_rows = denseInput.featureMat.size();    // number of rows
    csrX.number_cols = denseInput.featureMat[0].size() ;
    csrX.nnz = nnz ;
	std::cout << "number of samples: " << denseInput.featureMat.size() << 
	", nnz: " << nnz << std::endl;
	return csrX;
}    



void ReadDenseInput(char *DataFile, DenseData *data) 
{

    FILE *infp;
    infp = fopen(DataFile, "r");
    if (infp == NULL) {
        fprintf(stderr, "Error: could not open input file\n");
        fprintf(stderr, "Exiting ...\n");
        exit(1);
    }
 
    char *linebuf = NULL ; 
    size_t readlen = 0;

    char *linecp;
    char *tokpt; 
    
    std::vector<double> temprow;
    int k;
    while ( getline(&linebuf, &readlen, infp) != -1  ) {
        
        linecp = strdup(linebuf);
        tokpt = strtok(linecp," , \t \n");
        k=0;
        while (tokpt != NULL) {

            if (k==0) {
                data->label.push_back(atoi(tokpt));
            } else {
                temprow.push_back(atof(tokpt));
            }
            k++;
            tokpt = strtok(NULL," \t \n");

        }
        data->featureMat.push_back(temprow) ; 
        temprow.clear() ; 
    }

    fclose(infp);

    if (linebuf)
        free(linebuf);
   
}    


