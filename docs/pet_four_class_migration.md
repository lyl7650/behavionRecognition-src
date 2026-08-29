# Pet four-class labels

The active pet normal-behavior labels, in fixed class-index order, are:

1. `eat`
2. `drink`
3. `active`
4. `resting`

New pet training data, teacher/student checkpoints, automatic pre-labeling, and offline evaluation must use this order. The former `other` label is retired.

Historical directories that still contain `other` are not used by the new pipeline. They must be re-split into the four labels before being used for a new training snapshot.

The regular workflow is now:

`1_scaffold_dirs.bat` → `2a_presort_label.bat` → `2b_presort_review.bat` → `2c_presort_apply.bat` → `3_run_crop.bat` → `4_run_train_distill.bat` → `5_run_test.bat`
