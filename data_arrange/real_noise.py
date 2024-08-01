import glob
import shutil
import os

source_dir = '/NAS_liujie/liujie/ds004332-download'
target_dir = '/NAS_liujie/liujie/Real_Noise'

all_sub = glob.glob(source_dir + '/sub-**')

for sub in all_sub:
    case_name = sub.split('/')[-1]

    os.makedirs(target_dir + '/gt', exist_ok=True)
    os.makedirs(target_dir + '/image', exist_ok=True)
    source_path = sub + '/anat/' + case_name + '_acq-mpragepmcoff_rec-wre_run-03_T1w.nii'
    if os.path.exists(source_path):
        destination_path = target_dir + '/gt/' + case_name + '.nii'
        shutil.copy2(source_path, destination_path)

        source_path = sub + '/anat/' + case_name + '_acq-mpragepmcoff_rec-wore_run-03_T1w.nii'
        destination_path = target_dir + '/image/' + case_name + '.nii.gz'
        shutil.copy2(source_path, destination_path)

        print(case_name + ' finished.')
