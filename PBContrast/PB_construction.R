library(compositions)

# Functions to calculate the original PCA PBs are adapted from https://github.com/NesrstovaV/PLS-PBs/tree/main

fBalChipman<-function(C,angle=TRUE){
  # given a coda set C
  # This function returns the balance with D parts
  # that is close (angle) to the first PC
  # If "angle=FALSE" the Max the var of scores
  #
  # columns
  col<-dim(C)[2]
  nbal<-col-1
  # clr-transfo
  clrC<-log(C) - rowMeans(log(C))
  
  # PCs
  #
  pcClr<-prcomp(clrC)
  #first PC: PC1
  pcClr1<-pcClr$rotation[,1]
  
  balsig<-sign(pcClr1)
  bal<-matrix(0,nbal,col)
  colnames(bal)<-colnames(C)
  # balances associated to the PCs
  # first bal
  bal[1,pcClr1==max(pcClr1)]<-1
  bal[1,pcClr1==min(pcClr1)]<--1
  numbal=1
  
  # other bal
  if (col>2){
    numbal=numbal+1
    while (numbal<col){
      bal[numbal,]<-bal[numbal-1,]
      useonly<-(bal[numbal-1,]==0)
      bal[numbal,abs(pcClr1)==max(abs(pcClr1[useonly]))]<-balsig[abs(pcClr1)==max(abs(pcClr1[useonly]))]
      numbal=numbal+1
    }#end while
  }#end if
  
  # coefficients & angle
  VarSBP<-rep(0,nbal)
  for (f in 1:nbal) {
    den<-sum(bal[f,]==-1)
    num<-sum(bal[f,]==1)  
    bal[f,bal[f,]==1]<-sqrt(den/((den+num)*num))
    bal[f,bal[f,]==-1]<--sqrt(num/((den+num)*den))
    # variance of the balance
    VarSBP[f]<-abs(sum(bal[f,]*pcClr1))
  }
  # log-trasnform
  lC<-as.matrix(log(C))
  mvar=var(as.vector(lC%*%bal[VarSBP==max(VarSBP),]))
  
  if (!angle) {
    
    # calculate variance in the balance direction
    VarSBP<-rep(0,nbal)
    
    for (i in 1:nbal)
    {
      Proj<-as.vector(lC%*%(bal[i,]))
      VarSBP[i]<-var(Proj)
    }# end for
    mvar=max(VarSBP)
  }# end if
  
  # return results
  return(list(bal=bal[VarSBP==max(VarSBP),],varbal=mvar))
}

fBPUpChi<-function(Yp,b){
  
  # given coda set Yp 
  # and given a balance with some zero
  # return list of PARENT principal balances basis
  # that maximizises the variance
  # searching by the NO-FULL {0, -1,+1} and COMPLETING the
  # SBP using a loop (UP) scheme by the CHIPMAN procedure
  
  npart=ncol(Yp)
  nbal=ncol(Yp)-1
  # to save balances and variances
  Bal=c()
  VarB=c()
  
  usezero<-sum(b==0)
  
  #log-transfo data
  lYp=as.matrix(log(Yp))
  # while it is not the full balance go up
  k=0
  while (usezero>0){
    # new balance
    k<-k+1
    # non-zero in the will be in the denominator
    den<-sum(b!=0)
    # for only one zero we get the full
    if (usezero==1){
      
      b[b!=0]<--sqrt(1/((den+1)*den))
      b[b==0]<-sqrt(den/(den+1))
      
      VarB<-cbind(VarB,var(as.vector(lYp%*%b)))
      Bal<-rbind(Bal,b)
      usezero<-0
    }
    # for more than one zero we explore other {0,+1} combinations 
    else{
      # create the combination by CHIPMAN procedure
      # search the maximum balance
      clrC<-log(Yp[,b==0]) - rowMeans(log(Yp[,b==0]))
      # PCs
      #
      pcClr<-prcomp(clrC)        
      #first PC: PC1
      bx<-pcClr$rotation[,1]
      
      #
      # look for change of sign
      if (abs(min(bx))>max(bx)){bx<--bx}
      # force zeros to the other sign
      bx[bx<0]<-0
      # matrix of {0,+1} possibilities
      M<-matrix(0,sum(bx>0),length(bx))
      # sort
      bxsort<-sort(bx,decreasing = TRUE,index.return=TRUE)
      # index
      col<-bxsort$ix
      # create M
      for (i in 1:nrow(M)){
        M[i,col[1:i]]<-abs(bx[col[1:i]])
      }
      # sign
      M<-sign(M)
      
      # create a balance
      balax<-b
      # old non-zero to denominator
      balax[b!=0]<--1
      # search the max variance
      VarSBPx<-matrix(0,1,nrow(M))
      balsx<-c()
      for (i in 1:nrow(M))
      {
        # take one possibility
        balax[b==0]<-M[i,]
        # create the coefficients
        num<-sum(balax==1)
        balax[balax==1]<-sqrt(den/((den+num)*num))
        balax[balax==-1]<--sqrt(num/((den+num)*den))
        balsx=rbind(balsx,balax)
        
        VarSBPx[i]<-var(as.vector(lYp%*%balax))
      }
      VarB=cbind(VarB,max(VarSBPx))
      Bal=rbind(Bal,balsx[VarSBPx==VarB[k],])
      
      rm(M)
      usezero<-sum(Bal[k,]==0)
      b<-Bal[k,]
      # end else GO UP
    }
    
    # end GO UP while
  }
  # return results
  return(list(bal=Bal,varbal=VarB))
  # end function
}

fBPMaxOrthNewChip<-function(Y,angle=TRUE){
  
  # recursion: given a coda set Y
  # return list of principal balances basis
  # that maximizises the variance
  # searching by the NO-FULL and COMPLETING the
  # SBP using loop (UP: fBUpChi.r) and recursive (DOWN) schemes
  # both based on Chipman procedure
  
  numpart=ncol(Y)
  numbal=ncol(Y)-1
  B=c()
  #B=matrix(0,numbal,numpart) to save balances
  V=c()
  #V=matrix(0,1,numbal) to save variances
  
  #first optimal in data set Y
  res<-fBalChipman(Y,angle=angle)
  B<-res$bal
  V<-res$varbal
  # if necessary GO UP to complete
  if (sum(B==0)>0){  
    res<-fBPUpChi(Y,B)
    B=rbind(B,res$bal)
    V=cbind(V,res$varbal)
  }
  # control number of balances added
  if (is.vector(B)) B<-matrix(B,1,length(B)) 
  numbaladd<-nrow(B)-1
  
  ### GO DOWN THE CURRENt LIST AND THE FIRST
  ## first go down from the first optimal balance
  
  usenum<-(B[1,]>0)
  useden<-(B[1,]<0)
  # GO DOWN from numerator of the first optimal balance
  if(sum(usenum)>1){
    resP<-fBPMaxOrthNewChip(Y[,usenum],angle=angle)
    Bx<-matrix(0,length(resP$varbal),numpart)
    Bx[,usenum]<-resP$bal
    B<-rbind(B,Bx)
    V<-cbind(V,resP$varbal)
  }# end if
  # GO DOWN from denominator of the first optimal balance
  if(sum(useden)>1){
    resP<-fBPMaxOrthNewChip(Y[,useden],angle=angle)
    Bx<-matrix(0,length(resP$varbal),numpart)
    Bx[,useden]<-resP$bal
    B<-rbind(B,Bx)
    V<-cbind(V,resP$varbal)
  }# end if
  
  # REVISIT list of balances added GO UP so as to complete the SBP if necessary GO DOWN by the POSITIVE
  
  if (numbaladd > 0){
    for (k in 2:(1+numbaladd)){
      usepos=(B[k,]>0)
      if (sum(usepos)>1) {
        resP<-fBPMaxOrthNewChip(Y[,usepos],angle=angle)
        Bx<-matrix(0,length(resP$varbal),numpart)
        Bx[,usepos]<-resP$bal
        B<-rbind(B,Bx)
        V<-cbind(V,resP$varbal)
      }#end if2
    }# end for
  }# end if1
  
  # return results
  #
  V<-as.matrix(V,1,length(V))
  #
  return(list(bal=B,varbal=V))
}

fBalChip<-function(Xcoda){
  # given  a coda set Xcoda
  # this function calls fBPMaxOrthNewChip for a searching using the algortihm for Constrained PCs
  # and it returns ALL balances with D parts
  # that maximizes the variance
  # The balances are sorted by the percentatge of variance
  #
  # Returns a list: balances and variance
  # Bres= balances
  # Vres= variance of balances
  
  numbal=ncol(Xcoda)-1
  # call the recursive function
  res<-fBPMaxOrthNewChip(Xcoda, angle = F)
  Bres<-res$bal
  balname<-paste("bal",1:nrow(Bres),sep="")  
  rownames(Bres)<-balname
  colnames(Bres)<-colnames(Xcoda)
  Vres<-res$varbal
  #
  # sort by expl var
  vopt<-res$varbal
  # sort variance
  vsopt<-sort(vopt,decreasing = TRUE,index.return=TRUE)
  #
  # assign variance explained already ordered
  Vres<-vsopt$x
  #
  # assign balances same order
  Bres<-Bres[vsopt$ix,]
  #  
  # return results: balances and variances
  return(list(bal=Bres,varbal=Vres)) 
}

get_pvalue_summary<-function(y, C, bal, bmd_site, indiv = TRUE){
  p_cache<-c()
  bal_cache<-c()
  data_run<-cbind(y[, bmd_site], C)
  colnames(data_run)[1]<-bmd_site
  if(indiv == T){
    for(i in 1:ncol(bal)){
      tempdata_run<-cbind(data_run, bal[, i])
      colnames(tempdata_run)[ncol(data_run)+1]<-colnames(bal)[i]
      lmmodel_temp<-lm(paste(bmd_site, "~.", sep = ""), data = tempdata_run)
      temp_summary<-summary(lmmodel_temp)$coefficients[,4]
      p_cache<-c(p_cache, temp_summary[grep("bal", names(temp_summary))])
      bal_cache<-c(bal_cache, names(temp_summary)[grep("bal", names(temp_summary))])
    }
    p_cache_adj<-p.adjust(p_cache, method = "BH")
    pvalue_summary<-data.frame(bal = bal_cache, fdr_pvalue = p_cache_adj)
  }else{
    tempdata_run<-cbind(data_run, bal)
    lmmodel_temp<-lm(paste(bmd_site, "~.", sep = ""), data = tempdata_run)
    temp_summary<-summary(lmmodel_temp)$coefficients[,4]
    p_cache<-temp_summary[grep("bal", names(temp_summary))]
    bal_cache<-names(temp_summary)[grep("bal", names(temp_summary))]
    pvalue_summary<-data.frame(bal = bal_cache, pvalue = p_cache)
  }
  return(pvalue_summary)
}

get_important_microbes<-function(Xtrain, Ctrain, ytrain, pheno_list, nbalance = ncol(Xtrain)-1, indiv = TRUE, pval_threshold = 0.1){
  print(nbalance)
  imp_microbes_list<-list()
  Xtrainclr <-log(Xtrain) - rowMeans(log(Xtrain))
  Xtrain_C <- as.data.frame(compositions::clrInv(Xtrainclr))
  for(i in pheno_list){
    temp_train_pca <- fBalChip(Xtrain)
    temp_balance_train <- temp_train_pca$bal[c(1:nbalance),]
    temp_soucin <- as.matrix(log(Xtrain_C))%*%t(temp_balance_train)
    temp_soucin_data <- as.data.frame(temp_soucin)
    
    temp_pvalue_cache <- get_pvalue_summary(ytrain, Ctrain, temp_soucin_data, i, indiv)
    if(indiv == T){
      temp_imp_bal <- temp_pvalue_cache$bal[temp_pvalue_cache$fdr_pvalue<pval_threshold]
    }else{
      temp_imp_bal <- temp_pvalue_cache$bal[temp_pvalue_cache$pvalue<pval_threshold]
    }
    temp_balance_save <- temp_balance_train[temp_imp_bal,]
    
    temp_microbes_cache <- c()
    if(length(temp_imp_bal)>1){
      for(j in temp_imp_bal){
        temp_microbes_cache <- c(temp_microbes_cache, 
                                 colnames(temp_balance_train)[temp_balance_train[j,]!=0])
      }
      temp_microbes_table <- table(temp_microbes_cache)
      imp_microbes_cache <- names(temp_microbes_table[temp_microbes_table>2])
      imp_microbes_list[[i]] <- list(imp_microbes = imp_microbes_cache,
                                     imp_bal_names = temp_imp_bal,
                                     imp_bal_save = temp_balance_save)
    }else if(length(temp_imp_bal)==1){
      imp_microbes_cache <- colnames(temp_balance_train)[temp_balance_train[temp_imp_bal[1],]!=0]
      imp_microbes_list[[paste(i,"_only_one_bal", sep = "")]] <- list(imp_microbes = imp_microbes_cache,
                                                                      imp_bal_names = temp_imp_bal,
                                                                      imp_bal_save = temp_balance_save)
    }
  }
  return(imp_microbes_list)
}

get_balance_scores<-function(C, bal){
  C_clr <-log(C) - rowMeans(log(C))
  C2 <- as.data.frame(compositions::clrInv(C_clr))
  if(is.vector(bal)){
    bal_scores <- as.data.frame(as.matrix(log(C2))%*%t(as.matrix(bal)))
  }else{
    bal_scores <- as.data.frame(as.matrix(log(C2))%*%t(bal))
  }
  return(bal_scores)
}

get_matched_balance_scores<-function(X_tu, X_te, bal, pb_method, pheno_list, data_path, file_names, condition = "_"){
  setwd(paste(c(data_path, "train_test_split/"), collapse = ""))
  tune_id<-read.csv("subject_id_tu.csv", h = T)
  tune_bal_score_cache<-list()
  for(i in pheno_list){
    temp_bal<-bal[[i]]$imp_bal_save
    tune_bal_score_cache[[i]]<-get_balance_scores(X_tu, temp_bal)
    test_bal_scores<-get_balance_scores(X_te, temp_bal)
    write.csv(tune_bal_score_cache[[i]], file = paste(c(pb_method, "_", i, condition, "selected_balance_scores_tu.csv"), collapse = ""), row.names = F)
    write.csv(test_bal_scores, file = paste(c(pb_method, "_", i, condition, "selected_balance_scores_te.csv"), collapse = ""), row.names = F)
  }
  for(folder in file_names){
    setwd(paste(data_path, folder, sep = ""))
    train_id<-read.csv(paste(c("subject_id_tr_", folder, ".csv"), collapse = ""), h = T)
    valid_id<-read.csv(paste(c("subject_id_val_", folder, ".csv"), collapse = ""), h = T)
    matched_id_train<-match(train_id$sampleID, tune_id$sampleID)
    matched_id_valid<-match(valid_id$sampleID, tune_id$sampleID)
    for(i in pheno_list){
      write.csv(tune_bal_score_cache[[i]][matched_id_train,], file = paste(c(pb_method, "_", i, condition, "selected_balance_scores_tr_", folder, ".csv"), collapse = ""), row.names = F)
      write.csv(tune_bal_score_cache[[i]][matched_id_valid,], file = paste(c(pb_method, "_", i, condition, "selected_balance_scores_val_", folder, ".csv"), collapse = ""), row.names = F)
    }
  }
}

library(zCompositions)
setwd("~/data_divisions/train_test_split")
x_testout_tu<-read.csv("microbe_comp_tu.csv", h=T)
y_testout_tu<-read.csv("bmd_tu.csv", h=T)
x_testout_te<-read.csv("microbe_comp_te.csv", h=T)
y_testout_te<-read.csv("bmd_te.csv", h=T)
c_testout_tu<-read.csv("clinical_var_tu.csv", h=T)
c_testout_te<-read.csv("clinical_var_te.csv", h=T)

microbes_mask1<-read.csv("microbe_names_prev_filtered.csv", h = T)
x_testout_tu1<-x_testout_tu[,microbes_mask1$species]
x_testout_te1<-x_testout_te[,microbes_mask1$species]

min_nz_tu1<-min(x_testout_tu1[x_testout_tu1 > 0], na.rm = TRUE)
min_nz_te1<-min(x_testout_te1[x_testout_te1 > 0], na.rm = TRUE)

X_imp_tu1<-multRepl(
  x_testout_tu1,
  label=0,                     # your zeros are the censored values
  dl=rep(min_nz_tu1, ncol(x_testout_tu1)), # same DL per column → impute to imp_val
  frac=0.5,                     # so imputed value = 1 * dl = imp_val
  closure=100,                   # keep your original scale / row totals
  z.delete=FALSE
)

X_imp_te1<-multRepl(
  x_testout_te1,
  label=0,                     # your zeros are the censored values
  dl=rep(min_nz_te1, ncol(x_testout_te1)), # same DL per column → impute to imp_val
  frac=0.5,                     # so imputed value = 1 * dl = imp_val
  closure=100,                   # keep your original scale / row totals
  z.delete=FALSE
)

#all.equal(x_testout_tu2[x_testout_tu2 > 0], X_imp_tu2[x_testout_tu2 > 0])

X_imp_tu1<-as.data.frame(compositions::clo(X_imp_tu1) * 100)
X_imp_te1<-as.data.frame(compositions::clo(X_imp_te1) * 100)

pheno_list<-colnames(y_testout_tu)
important_microbes_list<-get_important_microbes(X_imp_tu1, c_testout_tu, y_testout_tu, pheno_list, indiv = F, pval_threshold = 0.05)
setwd("~/data_divisions/train_test_split/selected_balances")
#write.csv(data.frame(species = important_microbes_list$NECK_BMD[["imp_microbes"]]), file = "PCA_PB_prevfiltered_fneck_only_selected_species.csv", row.names = F)
write.csv(data.frame(bals = important_microbes_list$NECK_BMD$imp_bal_names), file = "PCA_PB_prevfiltered_fneck_only_selected_bal_names.csv", row.names = F)
write.csv(important_microbes_list$NECK_BMD$imp_bal_save, file = "PCA_PB_prevfiltered_fneck_only_selected_bals.csv")

#write.csv(data.frame(species = important_microbes_list$HTOT_BMD[["imp_microbes"]]), file = "PCA_PB_prevfiltered_htot_only_selected_species.csv", row.names = F)
write.csv(data.frame(bals = important_microbes_list$HTOT_BMD$imp_bal_names), file = "PCA_PB_prevfiltered_htot_only_selected_bal_names.csv", row.names = F)
write.csv(important_microbes_list$HTOT_BMD$imp_bal_save, file = "PCA_PB_prevfiltered_htot_only_selected_bals.csv")

#write.csv(data.frame(species = important_microbes_list$spine_total_bmd[["imp_microbes"]]), file = "PCA_PB_prevfiltered_spine_only_selected_species.csv", row.names = F)
write.csv(data.frame(bals = important_microbes_list$spine_total_bmd$imp_bal_names), file = "PCA_PB_prevfiltered_spine_only_selected_bal_names.csv", row.names = F)
write.csv(important_microbes_list$spine_total_bmd$imp_bal_save, file = "PCA_PB_prevfiltered_spine_only_selected_bals.csv")

#write.csv(data.frame(species = important_microbes_list$R_13_BMD[["imp_microbes"]]), file = "PCA_PB_prevfiltered_R13_only_selected_species.csv", row.names = F)
write.csv(data.frame(bals = important_microbes_list$R_13_BMD$imp_bal_names), file = "PCA_PB_prevfiltered_R13_only_selected_bal_names.csv", row.names = F)
write.csv(important_microbes_list$R_13_BMD$imp_bal_save, file = "PCA_PB_prevfiltered_R13_only_selected_bals.csv")

data_path<-"~/data_divisions/"
file_names<-paste0("tune_", seq(1,10))
get_matched_balance_scores(X_imp_tu1, X_imp_te1, important_microbes_list, "PCA_PB", pheno_list, data_path, file_names, "_prevfiltered_species_")
