from data_pkg import data_fns as df

# The following commands will take the videos in trafficdb and convert them into images
# According to the database recommendation, eval_0, eval_1, eval_2, eval_3 sets are generated
# eval_0, eval_1, eval_2, eval_3 are different train-test splits of the same database.

df.save_train_test_from_db(0)
df.save_train_test_from_db(1)
df.save_train_test_from_db(2)
df.save_train_test_from_db(3)

