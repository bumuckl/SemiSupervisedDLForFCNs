% Creates a network architecture
%
% Author: Christoph Baur

clear;
close all;
run ../../../Setup.m

% Options
CNN_opts;

% Override option
options.debug = true;
options.data.domains = {'D'};
options.data.patients = {};
options.test.epoch = 'last'; %'10' or 'last'
options.test.embeddings = false;
options.test.tsne = false;
options.test.embeddingDomains = {'A', 'B', 'C', 'D'};
options.test.pca = false;
options.test.pca3D = false;
options.test.savefig = true;
options.test.savePredictedVolumes = false;
options.test.saveSliceVisualizations = true;
options.test.savePredictedBinaryVolumes = false;
options.test.plotROC = false;
options.test.models = {
%     [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'],
    [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainD_c128x128_l1e-06_b6/'],
%     
%     [options.train.expDir_prefix '/MSSemiSupervised/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_NCC_patient5-6/'],
%     [options.train.expDir_prefix '/MSUpperbound/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_finetuneAB/'],
%     
%     [options.train.expDir_prefix '/MSSemiSupervised/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAC_NCC_patient8-9/'],
%     [options.train.expDir_prefix '/MSUpperbound/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_finetuneAC/'],
%     
%     [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'],
%    [options.train.expDir_prefix '/MSSemiSupervised/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAD_NCC_patient11-12/'],
%     [options.train.expDir_prefix '/MSUpperbound/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_finetuneAD/'],
      
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/'],
%     
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/'],
%     
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/'],
%     
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\50-50\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/'],
    
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\80-20\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/'],
%     
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsEuclidean\distAware\MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda0.001_m1000_conv_u0cd_hadsell_lu1_SSembeddingLoss_finetuneAB_distAware_full2small2000/']

%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small20/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small100/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small200/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small500/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small1000/'],
%     [options.train.expDir_prefix '/MSNumEmbeddingsCosine\80-20\2\MSSEG_UNET_FbetaLossWithUpdate_Domain_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_cosine_finetuneAB_distAware_full2small2000/']
};
%options.train.expDir = [options.train.expDir_prefix '/MSUpperbound/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_finetuneAC/'];
%options.train.expDir = [options.train.expDir_prefix '/MSSemiSupervised/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6_lambda1_m1_conv_u0cd_cosine_lu1_SSembeddingLoss_finetuneAC_NCC_patient8-9/'];
%options.train.expDir = [options.train.expDir_prefix '/MSBaseline/MSSEG_UNET_FbetaLossWithUpdate_DomainA_c128x128_l1e-06_b6/'];


% Do the test
for m=1:length(options.test.models)
    options.train.expDir = options.test.models{m};
    [ Eval, embeddings ] = fn_test(options);
end