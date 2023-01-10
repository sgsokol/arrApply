// [[Rcpp::depends(RcppArmadillo)]]
#include <RcppArmadillo.h>
using namespace arma;

#include <Rcpp.h>
using namespace Rcpp;

// define integers for function names
enum array_act { a_sum, a_prod, a_all, a_any, a_min, a_max, a_mean, a_median, a_sd, a_var, a_norm, a_trapz, a_normalise, a_cumsum, a_cumprod, a_multv, a_divv, a_addv, a_subv, a_diff, a_range, a_conv, a_quantile };

// result of a call on a vector is a real scalar
#define make_call(act) \
        for (std::size_t isl=0; isl < work.n_slices; ++isl) {\
            res(span::all,isl)=arma::act(work.slice(isl), 1);\
        }
// result of a call on a vector is a logical scalar
#define make_call_l(act) \
        for (std::size_t isl=0; isl < work.n_slices; ++isl) {\
            lres(span::all,isl)=arma::act(work.slice(isl), 1);\
        }
// result of a call on a vector is a real vector
#define make_call_a(act) \
        for (std::size_t isl=0; isl < work.n_slices; ++isl) {\
            cres.slice(isl)=arma::act(work.slice(isl), 1);\
        }

// numerical reduced result
#define CASE(act) \
case a_##act: \
  make_call(act);\
  break;

// logical result
#define CASE_L(act) \
case a_##act: \
  make_call_l(act);\
  break;

// numerical mapped result
#define CASE_A(act) \
case a_##act: \
  cres=work; \
  make_call_a(act); \
  break;

// [[Rcpp::interfaces(r)]]
//' High Performance Variant of apply()
//'
//'  High performance variant of apply() for a fixed set of functions.
//'  Considerable speedup obtained by this implementation is a trade-off for universality, user defined
//'  functions cannot be used with arrApply. However, about 20 most currently employed
//'  functions are available for usage. They can be divided in three types: \itemize{
//'    \item reducing functions (like mean(), sum() etc., giving a scalar when applied to a vector);
//'    \item mapping function (like normalise(), cumsum() etc., giving a vector of the same length
//'        as the input vector)
//'    \item and finally, vector reducing function (like diff() which produces
//'        result vector of a length different from the length of input vector).
//'  }
//'  Optional or mandatory additional arguments required by some functions
//'  (e.g. norm type for norm() or normalise() functions) can be
//'  passed as named arguments in '...'.
//' 
//' The following functions can be used as argument 'fun' (brackets
//' [] indicate additional parameters that can be passed in '...'): \itemize{
//'     \item reducing functions: \itemize{
//'         \item sum()
//'         \item prod()
//'         \item all()
//'         \item any()
//'         \item min()
//'         \item max()
//'         \item mean()
//'         \item median()
//'         \item sd() [norm_type]
//'         \item var() [norm_type]
//'         \item norm() [p],
//'         \item trapz() [x] (trapezoidal integration with respect to spacing in x,
//'            if x is provided, otherwise unit spacing is used)
//'         \item range();
//'       }
//'     \item mapping functions:\itemize{
//'         \item normalise() [p]
//'         \item cumsum()
//'         \item cumprod()
//'         \item multv() [v]
//'            (multiply a given dimension by a vector v, term by term)
//'         \item divv() [v]
//'            (divide by a vector v)
//'         \item addv() [v] (add a vector v)
//'         \item subv() [v] (subtract
//'            a vector v);
//'         }
//'     \item vector reducing/augmenting function:\itemize{
//'         \item diff() [k]
//'         \item conv() [v, shape] (convolve with vector v; shape="full" is equivalent
//'            to R's \code{convolve(..., rev(v), type="open")}).
//'         \item quantile() [p] (calculate quantiles corresponding to probabilities p;
//'            equivalent to R's \code{quantile(..., probs=p, type=8)}).
//'        }
//'     }
//' 
//' RcppArmadillo is used to do the job in very fast way but it comes at price
//' of not allowing NA in the input numeric array.
//' Vectors are allowed at input. They are considered as arrays of dimension 1.
//' So in this case, \code{idim} can only be 1.
//' NB. Here, range() is different from R version of the homonym function.
//'      In Armadillo, when applied to a vector, it returns a scalar max-min,
//'      while in R, it return a 2-component vector (min, max).
//' 
//' 
//' @param arr numeric array of arbitrary dimension
//' @param idim integer, dimension number along which a function must be applied
//' @param fun character string, function name to be applied
//' @param ... additional named parameters. Optional parameters can be helpful for
//'    the following functions:\itemize{
//'       \item sd(), var() [norm_type: 0 normalisation using N-1 entries (default);
//'          1 normalisation using N entries];
//'       \item norm() [p: integer >= 1 (default=2) or one of "-inf", "inf", "fro".]
//'       \item normalise() [p: integer >= 1, default=2]
//'       \item diff() [k: integer >= 1 (default=1) number of recursive application of diff().
//'          The size of idim-th dimension will be reduced by k.]
//'       \item trapz() [x: numerical vector of the same length as idim-th size of arr]
//'    }
//'    Mandatory parameter:\itemize{
//'       \item multv(), divv(), addv(), subv() [v: numerical vector of the same
//'          length as idim-th size of arr]
//'       \item quantile() [p: vector of probabilities in interval [0; 1]]
//'    }
//'
//' @return output array of dimension cut by 1 (the idim-th dimension
//'    will disappear for reducing functions) or of the same dimension
//'    as the input arr for mapping and vector reducing
//'    functions. For vector reducing functions, the idim-th dimension
//'    will be different from idim-th dimension of arr.
//'    The type of result (numeric or logical) depends on the function applied,
//'    logical for all() and any(), numerical -- for all other functions.
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
SEXP arrApply(NumericVector arr, unsigned int idim=1, std::string fun="sum", List dots=R_NilValue) {
    std::map<std::string, array_act> mapf;
    #define add_map(act) mapf[#act]=a_##act
    // populate the mapf
    // reducing functions
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
    add_map(norm);
    add_map(trapz);
    add_map(range);
    // mapping functions
    add_map(normalise);
    add_map(cumsum);
    add_map(cumprod);
    add_map(multv);
    add_map(divv);
    add_map(addv);
    add_map(subv);
    // vector reducing functions
    add_map(diff);
    add_map(conv);
    add_map(quantile);
  
#define NBUF 512
    char buf[NBUF];
    if (mapf.count(fun) == 0) {
        snprintf(buf, NBUF, "arrApply: fun='%s' is not in the list of applicable functions", fun.c_str());
        stop(buf);
    }
    array_act aact=mapf[fun];
    std::vector<array_act> mvact={a_normalise, a_cumsum, a_cumprod, a_multv, a_divv, a_addv, a_subv, a_diff, a_conv, a_quantile}; // mapping or vector reducing acts
    bool mv_res=std::find(mvact.begin(), mvact.end(), aact) != mvact.end();
    uvec d;
    // optional parameters
    unsigned int p=2, norm_type=0, k=1;
    std::string pch="", shape="full";
    vec x;
    // mandatory parameter
    rowvec rowv;
    // auxiliary variables
    bool p_is_int=true, use_x=false;
    RObject robj;
    double pr;
    //Rcout << "mapf.count('" << fun << "')=" <<  mapf.count(fun) << std::endl;
    //for (auto it=mapf.cbegin(); it != mapf.cend(); ++it)
    //    Rcout << it->first << "=>\t" << it->second << std::endl;
    // get args from dots
    switch(aact) {
        case a_sd:
        case a_var:
            if (dots.containsElementNamed("norm_type")) {
                //get "norm_type" in dots
                norm_type=as<unsigned int>(dots["norm_type"]);
                if (norm_type != 0 && norm_type != 1) {
                    snprintf(buf, NBUF, "arrApply: optional norm_type must be 0 or 1, instead got %d.", norm_type);
                    stop(buf);
                }
            }
        break;
        case a_norm:
        case a_normalise:
            if (dots.containsElementNamed("p")) {
                //get "p" in dots
                robj=dots["p"];
                switch(robj.sexp_type()) {
                    case INTSXP:
                        p=as<unsigned int>(robj);
                    break;
                    case REALSXP:
                        pr=as<double>(robj);
                        if (std::abs(round(pr)-pr) < 1.e-10) {
                            p=round(pr);
                        } else {
                            snprintf(buf, NBUF, "arrApply: optional p must be integer >= 1, instead got %g.", pr);
                            stop(buf);
                        }
                    break;
                    case STRSXP:
                        pch=as<std::string>(robj);
                        p_is_int=false;
                    break;
                    default:
                        snprintf(buf, NBUF, "arrApply: unauthorized type for optional p (must be integer >= 1 or string), instead got SEXP type %d.", robj.sexp_type());
                        stop(buf);
                    break;
                }
                if (!p_is_int && !(pch == "-inf" || pch == "inf" || pch == "fro")) {
                    snprintf(buf, NBUF, "arrApply: optional p, when a string, must be one of '-inf', 'inf' or 'fro'. Instead, got '%s'.", pch.c_str());
                    stop(buf);
                } else if (p_is_int && p < 1) {
                    snprintf(buf, NBUF, "arrApply: optional p, when an integer, must be >= 1. Instead, got '%d'.", p);
                    stop(buf);
                }
            }
        break;
        case a_diff:
            if (dots.containsElementNamed("k")) {
                //get "k" in dots
                k=as<unsigned int>(dots["k"]);
                if (k <= 0) {
                    snprintf(buf, NBUF, "arrApply: optional k must be an integer >= 1, instead got %d.", k);
                    stop(buf);
                }
            }
        break;
        case a_multv:
        case a_divv:
        case a_addv:
        case a_subv:
            if (dots.containsElementNamed("v")) {
                rowv=as<rowvec>(dots["v"]);
            } else {
                snprintf(buf, NBUF, "Parameter v is mandatory for %s() function", fun.c_str());
                stop(buf);
            }
        break;
        case a_trapz:
            if (dots.containsElementNamed("x")) {
                x=as<vec>(dots["x"]);
                use_x=true;
            }
        break;
        case a_quantile:
            if (dots.containsElementNamed("p")) {
                x=as<vec>(dots["p"]);
            } else {
                snprintf(buf, NBUF, "Parameter p is mandatory for %s() function", fun.c_str());
                stop(buf);
            }
        break;
        case a_conv:
            if (dots.containsElementNamed("v")) {
                x=as<vec>(dots["v"]);
            } else {
                snprintf(buf, NBUF, "Parameter v is mandatory for %s() function", fun.c_str());
                stop(buf);
            }
            if (dots.containsElementNamed("shape")) {
                shape=as<std::string>(dots["shape"]);
            } else {
                shape=std::string("full");
            }
        break;
        default:
        ;
        break;
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
        snprintf(buf, NBUF, "arrApply: idim (%d) is out of range [1, %d]", idim, d.size());
        stop(buf);
    }
    
//Rprintf("ifun=%d\n");
    // reshape arr as a cube with sizes: before,do_it,after
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
    // dwork.print("dwork");
//Rprintf("yah4\n");
    
    cube work(arr.begin(), dwork[0], dwork[1], dwork[2], false);
//Rprintf("cube=%x\n", work.begin());
    bool logical_res=(aact == a_all || aact == a_any);
    mat res(dwork[0], dwork[2]);
    umat lres;
    cube cres;  // in case not reducing functions (like normalise()) are applied)
    if (logical_res) {
        lres.reshape(dwork[0], dwork[2]);
    }
//Rprintf("res=%x\n", res.begin());
    switch(aact) {
        CASE(prod);
        CASE(sum);
        CASE(min);
        CASE(max);
        CASE(mean);
        CASE(median);
        CASE(range);
        CASE_L(all);
        CASE_L(any);
        CASE_A(cumsum);
        CASE_A(cumprod);
        case a_sd:
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                res(span::all,isl)=stddev(work.slice(isl), norm_type, 1);
        break;
        case a_var:
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                res(span::all,isl)=var(work.slice(isl), norm_type, 1);
        break;
        case a_norm:
            if (p_is_int) {
                for (std::size_t isl=0; isl < work.n_slices; ++isl)
                    for (std::size_t ir=0; ir < res.n_rows; ir++)
                        res.at(ir,isl)=norm(work.slice(isl).row(ir), p);
            } else {
                for (std::size_t isl=0; isl < work.n_slices; ++isl)
                    for (std::size_t ir=0; ir < res.n_rows; ir++)
                        res.at(ir,isl)=norm(work.slice(isl).row(ir), pch.c_str());
            }
        break;
        case a_normalise:
            cres=work;
            if (p_is_int) {
                for (std::size_t isl=0; isl < work.n_slices; ++isl)
                    cres.slice(isl)=normalise(work.slice(isl), p, 1);
            } else {
                snprintf(buf, NBUF, "arrApply: for normalise() call, p is supposed to be integer, not a string.");
                stop(buf);
            }
        break;
        case a_diff:
//Rprintf("d1\n");
            cres.set_size(dwork[0], dwork[1]-std::min(k, dwork[1]), dwork[2]);
//Rprintf("d2\n");
//Rcout << "c.r=" << cres.n_rows << "; c.c=" << cres.n_cols << "; c.s=" << cres.n_slices << std::endl;
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                cres.slice(isl)=diff(work.slice(isl), k, 1);
//Rprintf("d3\n");
        break;
        case a_quantile:
            cres.set_size(dwork[0], x.size(), dwork[2]);
            for (std::size_t isl=0; isl < work.n_slices; ++isl)
                cres.slice(isl)=quantile(work.slice(isl), x, 1);
        break;
        case a_conv:
//Rprintf("d1\n");
            if (shape == "full")
                cres.set_size(dwork[0], dwork[1]+x.n_elem-1, dwork[2]);
            else if (shape == "same")
                cres.set_size(dwork[0], dwork[1], dwork[2]);
            else {
                snprintf(buf, NBUF, "arrApply: for conv(), if given, the parameter shape must be 'full' (default) or 'same', got '%s'", shape.c_str());
                stop(buf);
            }
//Rprintf("d2\n");
//Rcout << "c.r=" << cres.n_rows << "; c.c=" << cres.n_cols << "; c.s=" << cres.n_slices << std::endl;
        {
            rowvec mtmp;
            for (std::size_t isl=0; isl < work.n_slices; ++isl) {
                for (std::size_t ir=0; ir < work.n_rows; ir++) {
                    mtmp=conv(work.slice(isl).row(ir), x, shape.c_str());
                    cres.slice(isl).row(ir)=mtmp;
                }
            }
        }
//Rprintf("d3\n");
        break;
        case a_multv:
        case a_divv:
        case a_addv:
        case a_subv:
            cres=work;
            if (rowv.size() == work.n_cols) {
                switch (aact) {
                case a_multv:
                    for (std::size_t isl=0; isl < work.n_slices; ++isl)
                        cres.slice(isl).each_row() %= rowv;
                break;
                case a_divv:
                    for (std::size_t isl=0; isl < work.n_slices; ++isl)
                        cres.slice(isl).each_row() /= rowv;
                break;
                case a_addv:
                    for (std::size_t isl=0; isl < work.n_slices; ++isl)
                        cres.slice(isl).each_row() += rowv;
                break;
                case a_subv:
                    for (std::size_t isl=0; isl < work.n_slices; ++isl)
                        cres.slice(isl).each_row() -= rowv;
                break;
                default:
                break;
                }
            } else {
                snprintf(buf, NBUF, "arrApply: for multv() and alike, length(v) (%d) must be equal to dim(arr)[idim] (%d).", rowv.size(), work.n_cols);
                stop(buf);
            }
        break;
        case a_trapz:
            if (use_x) {
                if (x.size() == work.n_cols) {
                    for (std::size_t isl=0; isl < work.n_slices; ++isl)
                        res(span::all,isl)=trapz(x, work.slice(isl), 1);
                } else {
                    snprintf(buf, NBUF, "arrApply: for trapz(), length(x) (%d) must be equal to dim(arr)[idim] (%d).", x.size(), work.n_cols);
                    stop(buf);
                }
            } else {
                for (std::size_t isl=0; isl < work.n_slices; ++isl)
                    res(span::all,isl)=trapz(work.slice(isl), 1);
            }
        break;
        default:
           stop("arrApply: It cannot be, but unknown action is encountered.");
        break;
    }
    // rechape back the result
    if (!mv_res)
        d=join_cols(d.head(idim-1), d.tail(d.size()-idim));
//Rprintf("yah5\n");
    RObject vres;
    if (logical_res) {
        // logical actions
        LogicalVector v;
        v=lres;
        vres=wrap(v);
    } else if (mv_res) {
        // mapping/vec-reducing actions
        vres=wrap(cres);
        if (d.size()) {
            d[idim-1]=cres.n_cols;
            //d.print("dim final");
            vres.attr("dim")=IntegerVector(d.begin(), d.end());
        } else {
            vres.attr("dim")=R_NilValue;
        }
    } else {
        // reducing actions
        vres=wrap(res);
    }
    return vres;
}
