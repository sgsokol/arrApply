// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// define integers for function names
enum array_act { a_sum, a_prod, a_all, a_any, a_min, a_max, a_mean, a_median, a_sd, a_var };

#define make_call(act) \
        for (std::size_t isl=0; isl < work.n_slices; ++isl) {\
            res(span::all,isl)=arma::act(work.slice(isl), 1);\
        }
#define make_call_l(act) \
        for (std::size_t isl=0; isl < work.n_slices; ++isl) {\
            lres(span::all,isl)=arma::act(work.slice(isl), 1);\
        }
// numerical result
#define CASE(act) \
case a_##act: \
  make_call(act);\
  break;

// logical result
#define CASE_L(act) \
case a_##act: \
  make_call_l(act);\
  break;

// [[Rcpp::interfaces(r)]]
//' High Performance Variant of apply()
//'
//' High performance variant of apply() for a fixed set of functions.
//' However, a considerable speedup is a trade-off for universality.
//' Only the following functions can be applied: sum(), prod(), all(), any(), min(), max(),
//' mean(), median(), sd(), var().
//' 
//' RcppArmadillo is used to do the job very quickly but it commes at price
//' of not allowing NA in the input numeric array.
//' Vectors are allowed at input. They are considered as arrays of dimension 1.
//' So in this case, \code{idim} must be 1.
//' 
//' 
//' @param arr numeric array of arbitrary dimension
//' @param idim integer, dimension number along which a function must be applied
//' @param fun character string, function name to be applied
//'
//' @return output array of dimension cut by 1. Its type (nueric or logical)
//' depends on the function applied.
//' 
//' @examples
//'  arr=matrix(1:12, 3, 4)
//'  v1=arrApply(arr, 2, "mean")
//'  v2=rowMeans(arr)
//'  stopifnot(all(v1==v2))
//'  
//'  arr=array(1:24, dim=2:4) # dim(arr)=c(2, 3, 4)
//'  mat=arrApply(arr, 2, "prod") # dim(mat)=c(2, 4), the second dimension is cut out
//'  stopifnot(all(mat==apply(arr, c(1, 3), prod)))
//' 
//' @author Serguei Sokol <sokol at insa-toulouse.fr>
//' 
//' @export
// [[Rcpp::export]]
SEXP arrApply(NumericVector arr, int idim=1, std::string fun="sum") {
    
    std::map<std::string, array_act> mapf;
    #define add_map(act) mapf[#act]=a_##act
    // populate the mapf
    add_map(sum);
    add_map(prod);
    add_map(all);
    add_map(any);
    add_map(min);
    add_map(max);
    add_map(mean);
    add_map(median);
    add_map(sd);
    add_map(var);
    uvec d;
    char buf[512];
    
    if (mapf.find(fun) == mapf.end()) {
        sprintf(buf, "arrApply: fun '%s' cannot be applied", fun.data());
        stop(buf);
    }
    
//Rprintf("arr=%x\n", arr.begin());
    if (arr.hasAttribute("dim")) {
        IntegerVector dimi(arr.attr("dim"));
//Rprintf("yah2\n");
        d=as<uvec>(dimi);
//Rprintf("yah3\n");
    } else {
        d=uvec(1);
        d[0]=arr.size();
        idim=1;
    }
    if (idim < 1 || idim > d.size()) {
        sprintf(buf, "arrApply: idim (%d) is out of range [1, %d]", idim, d.size());
        stop(buf);
    }
    //ifun=std::distance(funs.begin(), std::find(funs.begin(), funs.end(), fun));
//Rprintf("ifun=%d\n");
    // reshape arr as a cube with dimensions: before,doit,after
    // if idim==1 then before=1, we do analogously for the last idim (after=1)
    uvec dwork(3, fill::zeros);
    if (idim == 1) {
        dwork[0]=1;
        dwork[1]=d[0];
        if (d.size() > 1) {
            dwork[2]=prod(d.tail(d.size()-idim));
        } else {
            dwork[2]=1;
        }
    } else if (idim == d.size()) {
        if (d.size() > 1) {
            dwork[0]=prod(d.head(idim-1));
        } else {
            dwork[0]=1;
        }
        dwork[1]=d.tail(1)[0];
        dwork[2]=1;
    } else {
        dwork[0]=prod(d.head(idim-1));
        dwork[1]=d[idim-1];
        dwork[2]=prod(d.tail(d.size()-idim));
    }
//Rprintf("yah4\n");
    
    cube work(arr.begin(), dwork[0], dwork[1], dwork[2], false);
//Rprintf("cube=%x\n", work.begin());
    bool logical_res=(fun=="all" | fun=="any");
    mat res(dwork[0], dwork[2]);
    umat lres;
    if (logical_res) {
        lres.reshape(dwork[0], dwork[2]);
    }
//Rprintf("res=%x\n", res.begin());
    switch(mapf[fun]) {
        CASE(prod);
        CASE(sum);
        CASE(min);
        CASE(max);
        CASE(mean);
        CASE(median);
        CASE_L(all);
        CASE_L(any);
        case a_sd:
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                res(span::all,isl)=stddev(work.slice(isl), 0, 1);
        break;
        case a_var:
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                res(span::all,isl)=var(work.slice(isl), 0, 1);
        break;
    }
    // rechape back the result
    d=join_cols(d.head(idim-1), d.tail(d.size()-idim));
//Rprintf("yah5\n");
    if (logical_res) {
        LogicalVector vres;
        vres=lres;
        if (d.size())
            vres.attr("dim")=IntegerVector(d.begin(), d.end());
        else
            vres.attr("dim")=R_NilValue;
        return vres;
    } else {
        RObject vres=wrap(res);
        if (d.size())
            vres.attr("dim")=IntegerVector(d.begin(), d.end());
        else
            vres.attr("dim")=R_NilValue;
        return vres;
    }
}
