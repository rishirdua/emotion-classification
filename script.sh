#generate raw files
echo "generating raw files"
python cPickleparser.py

python features_sampled.py

#genertate features

echo "generating bahsic features"
python features_bahsic.py "data/features_raw.dat" "data/labels_0.dat" "data/features_bahsic_0.dat" "data/features_normalized.dat"
python features_bahsic.py "data/features_raw.dat" "data/labels_1.dat" "data/features_bahsic_1.dat"
python features_bahsic.py "data/features_raw.dat" "data/labels_2.dat" "data/features_bahsic_2.dat"
python features_bahsic.py "data/features_raw.dat" "data/labels_3.dat" "data/features_bahsic_3.dat"

echo "generating rrt features"
echo "Run the matlab file to generate Recht and Rahimi Random Fourier features"

echo "generating downsampled features"
python features_sampled.py "data/features_sampled.dat"

# do a train-test split
python split_data.py "data/features_raw.dat" "data/labels_0.dat" "data/labels_test_0.dat"
python split_data.py "data/features_raw.dat" "data/labels_1.dat" "data/labels_test_1.dat"
python split_data.py "data/features_raw.dat" "data/labels_2.dat" "data/labels_test_2.dat"
python split_data.py "data/features_raw.dat" "data/labels_3.dat" "data/labels_test_3.dat"

#BAHSIC
echo "regression bahsic linear"
python regression_linear.py "data/features_bahsic_0.dat" "data/labels_0.dat" "predict/hsic_linear_0.dat 42"
python regression_linear.py "data/features_bahsic_1.dat" "data/labels_1.dat" "predict/hsic_linear_1.dat 42"
python regression_linear.py "data/features_bahsic_2.dat" "data/labels_2.dat" "predict/hsic_linear_2.dat 42"
python regression_linear.py "data/features_bahsic_3.dat" "data/labels_3.dat" "predict/hsic_linear_3.dat 42"

echo "regression bahsic bayesian"
python regression_bayesian.py "data/features_bahsic_0.dat" "data/labels_0.dat" "predict/hsic_bayesian_0.dat 42"
python regression_bayesian.py "data/features_bahsic_1.dat" "data/labels_1.dat" "predict/hsic_bayesian_1.dat 42"
python regression_bayesian.py "data/features_bahsic_2.dat" "data/labels_2.dat" "predict/hsic_bayesian_2.dat 42"
python regression_bayesian.py "data/features_bahsic_3.dat" "data/labels_3.dat" "predict/hsic_bayesian_3.dat 42"

echo "regression bahsic dtree"
python regression_dtree.py "data/features_bahsic_0.dat" "data/labels_0.dat" "predict/hsic_dtree_0.dat 42"
python regression_dtree.py "data/features_bahsic_1.dat" "data/labels_1.dat" "predict/hsic_dtree_1.dat 42"
python regression_dtree.py "data/features_bahsic_2.dat" "data/labels_2.dat" "predict/hsic_dtree_2.dat 42"
python regression_dtree.py "data/features_bahsic_3.dat" "data/labels_3.dat" "predict/hsic_dtree_3.dat 42"

echo "regression bahsic svr"
python regression_svr.py "data/features_bahsic_0.dat" "data/labels_0.dat" "predict/hsic_svr_0.dat 42"
python regression_svr.py "data/features_bahsic_1.dat" "data/labels_1.dat" "predict/hsic_svr_1.dat 42"
python regression_svr.py "data/features_bahsic_2.dat" "data/labels_2.dat" "predict/hsic_svr_2.dat 42"
python regression_svr.py "data/features_bahsic_3.dat" "data/labels_3.dat" "predict/hsic_svr_3.dat 42"

echo "regression bahsic logistic"
python regression_logistic.py "data/features_bahsic_0.dat" "data/labels_0.dat" "predict/hsic_logistic_0.dat 42"
python regression_logistic.py "data/features_bahsic_1.dat" "data/labels_1.dat" "predict/hsic_logistic_1.dat 42"
python regression_logistic.py "data/features_bahsic_2.dat" "data/labels_2.dat" "predict/hsic_logistic_2.dat 42"
python regression_logistic.py "data/features_bahsic_3.dat" "data/labels_3.dat" "predict/hsic_logistic_3.dat 42"

#rrt

echo "regression rrt linear"
python regression_linear.py "data/features_rrt.dat" "data/labels_0.dat" "predict/rrt_linear_0.dat"
python regression_linear.py "data/features_rrt.dat" "data/labels_1.dat" "predict/rrt_linear_1.dat"
python regression_linear.py "data/features_rrt.dat" "data/labels_2.dat" "predict/rrt_linear_2.dat"
python regression_linear.py "data/features_rrt.dat" "data/labels_3.dat" "predict/rrt_linear_3.dat"

echo "regression rrt bayesian"
python regression_bayesian.py "data/features_rrt.dat" "data/labels_0.dat" "predict/rrt_bayesian_0.dat"
python regression_bayesian.py "data/features_rrt.dat" "data/labels_1.dat" "predict/rrt_bayesian_1.dat"
python regression_bayesian.py "data/features_rrt.dat" "data/labels_2.dat" "predict/rrt_bayesian_2.dat"
python regression_bayesian.py "data/features_rrt.dat" "data/labels_3.dat" "predict/rrt_bayesian_3.dat"

echo "regression rrt dtree"
python regression_dtree.py "data/features_rrt.dat" "data/labels_0.dat" "predict/rrt_dtree_0.dat"
python regression_dtree.py "data/features_rrt.dat" "data/labels_1.dat" "predict/rrt_dtree_1.dat"
python regression_dtree.py "data/features_rrt.dat" "data/labels_2.dat" "predict/rrt_dtree_2.dat"
python regression_dtree.py "data/features_rrt.dat" "data/labels_3.dat" "predict/rrt_dtree_3.dat"

echo "regression rrt svr"
python regression_svr.py "data/features_rrt.dat" "data/labels_0.dat" "predict/rrt_svr_0.dat"
python regression_svr.py "data/features_rrt.dat" "data/labels_1.dat" "predict/rrt_svr_1.dat"
python regression_svr.py "data/features_rrt.dat" "data/labels_2.dat" "predict/rrt_svr_2.dat"
python regression_svr.py "data/features_rrt.dat" "data/labels_3.dat" "predict/rrt_svr_3.dat"

echo "regression rrt logistic"
python regression_logistic.py "data/features_rrt.dat" "data/labels_0.dat" "predict/rrt_logistic_0.dat"
python regression_logistic.py "data/features_rrt.dat" "data/labels_1.dat" "predict/rrt_logistic_1.dat"
python regression_logistic.py "data/features_rrt.dat" "data/labels_2.dat" "predict/rrt_logistic_2.dat"
python regression_logistic.py "data/features_rrt.dat" "data/labels_3.dat" "predict/rrt_logistic_3.dat"

#sampled 

echo "regression sampled linear"
python regression_linear.py "data/features_sampled.dat" "data/labels_0.dat" "predict/sampled_linear_0.dat"
python regression_linear.py "data/features_sampled.dat" "data/labels_1.dat" "predict/sampled_linear_1.dat"
python regression_linear.py "data/features_sampled.dat" "data/labels_2.dat" "predict/sampled_linear_2.dat"
python regression_linear.py "data/features_sampled.dat" "data/labels_3.dat" "predict/sampled_linear_3.dat"

echo "regression sampled bayesian"
python regression_bayesian.py "data/features_sampled.dat" "data/labels_0.dat" "predict/sampled_bayesian_0.dat"
python regression_bayesian.py "data/features_sampled.dat" "data/labels_1.dat" "predict/sampled_bayesian_1.dat"
python regression_bayesian.py "data/features_sampled.dat" "data/labels_2.dat" "predict/sampled_bayesian_2.dat"
python regression_bayesian.py "data/features_sampled.dat" "data/labels_3.dat" "predict/sampled_bayesian_3.dat"

echo "regression sampled dtree"
python regression_dtree.py "data/features_sampled.dat" "data/labels_0.dat" "predict/sampled_dtree_0.dat"
python regression_dtree.py "data/features_sampled.dat" "data/labels_1.dat" "predict/sampled_dtree_1.dat"
python regression_dtree.py "data/features_sampled.dat" "data/labels_2.dat" "predict/sampled_dtree_2.dat"
python regression_dtree.py "data/features_sampled.dat" "data/labels_3.dat" "predict/sampled_dtree_3.dat"

echo "regression sampled svr"
python regression_svr.py "data/features_sampled.dat" "data/labels_0.dat" "predict/sampled_svr_0.dat"
python regression_svr.py "data/features_sampled.dat" "data/labels_1.dat" "predict/sampled_svr_1.dat"
python regression_svr.py "data/features_sampled.dat" "data/labels_2.dat" "predict/sampled_svr_2.dat"
python regression_svr.py "data/features_sampled.dat" "data/labels_3.dat" "predict/sampled_svr_3.dat"

echo "regression sampled logistic"
python regression_logistic.py "data/features_sampled.dat" "data/labels_0.dat" "predict/sampled_logistic_0.dat"
python regression_logistic.py "data/features_sampled.dat" "data/labels_1.dat" "predict/sampled_logistic_1.dat"
python regression_logistic.py "data/features_sampled.dat" "data/labels_2.dat" "predict/sampled_logistic_2.dat"
python regression_logistic.py "data/features_sampled.dat" "data/labels_3.dat" "predict/sampled_logistic_3.dat"

