import glob
import shutil
import os

source_dir = '/NAS_liujie/liujie/ds004173-download/ds004173-download'
target_dir = '/NAS_liujie/liujie/MR-ART'

all_sub = glob.glob(source_dir + '/sub-**')

for sub in all_sub:
    case_name = sub.split('/')[-1]

    os.makedirs(target_dir + '/gt', exist_ok=True)
    source_path = sub + '/anat/' + case_name + '_acq-standard_T1w.nii.gz'
    destination_path = target_dir + '/gt/' + case_name + '.nii.gz'
    shutil.copy2(source_path, destination_path)

    os.makedirs(target_dir + '/image', exist_ok=True)
    source_path = sub + '/anat/' + case_name + '_acq-headmotion2_T1w.nii.gz'
    motion_type = '2'
    if not os.path.exists(source_path):
        source_path = sub + '/anat/' + case_name + '_acq-headmotion1_T1w.nii.gz'
        motion_type = '1'
    destination_path = target_dir + '/image/' + case_name + '.nii.gz'
    shutil.copy2(source_path, destination_path)

    print(case_name + ' finished. with motion ' + motion_type)
