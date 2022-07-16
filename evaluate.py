"""Compute metrics for trackers using ground-truth data."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import sys
from collections import OrderedDict
import glob
import logging
import os
from pathlib import Path
import motmetrics as mm
import zipfile
import shutil
import requests

evalMode = 'full'
fmt = 'mot15-2D'
exclude_id = False

# 错误字典
error_msg = {
    1: "Input File Error",
}


################ 
#jiangbaowei 7.11 define error message
error_message = dict()
error_message['Score'] = -1
error_message['MOTA'] = -1
error_message['MOTP'] = -1
error_message['status'] = -1
error_message['message'] = ''
error_message['unionKey'] = ''
################ 

def main():
    status = 0
    gt_root = sys.argv[1]
    result_root = sys.argv[2]
    score_root = sys.argv[3]
    # out_path = sys.argv[2]
    challenge_name = sys.argv[4]
    union_key = sys.argv[5]

    # read submit and answer file from first parameter
    # with open(in_param_path, 'r') as load_f:
    #     input_params = json.load(load_f)
    
    ################
    #Jiang baowei 7.12 take dir_name of submission
    submit_dir = os.path.dirname(result_root)
    logging.basicConfig(format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s',
                level=logging.DEBUG,
                filename=os.path.join(submit_dir,'evaluate.log'),
                filemode='a')
    logging.info('submit_dir is '+ submit_dir)
    ################

    # 标准答案路径
    # standard_path = input_params["fileData"]["standardFilePath"]
    standard_path = os.path.join(gt_root, challenge_name, 'mot_anno_full')
    print("Read standard from %s" % standard_path)

    # submit_path = input_params["fileData"]["userFilePath"]
    # submit_path = os.path.join(result_root, challenge_name, 'results.zip')
    submit_path = result_root
    print("Read user submit file from %s" % submit_path)
    out_path = score_root
    print("Save scores to :", out_path)
    
    ################
    #Jiang baowei 7.12 submission name mistake -> log and report error
    if not os.path.exists(os.path.join(submit_dir,'results.zip')):
        error_message['status'] = -1
        error_message['message'] = 'Cannot find results.zip file!'
        report_error_message(error_message,out_path,union_key)
        logging.info('Cannot find results.zip file!')
        return None
    ################
    
    # score_path = os.path.join(score_root, challenge_name, 'scores')
    # if not os.path.exists(score_path):
    #     os.makedirs(score_path)
    # out_path = os.path.join(score_path, 'eva_results.json')
   

    # 解压选手提交的压缩包
    unzip_file(submit_path, submit_dir)

    #loglevel = getattr(logging, 'INFO', None)
    #if not isinstance(loglevel, int):
    #    raise ValueError('Invalid log level: {} '.format(loglevel))
    #logging.basicConfig(level=loglevel, format='%(asctime)s %(levelname)s - %(message)s', datefmt='%I:%M:%S')
    
    ################
    #Jiang baowei 7.12 submission unzip mistake -> log and report error
    if not os.path.exists(os.path.join(submit_dir,'results')):
        error_message['status'] = -1
        error_message['message'] = 'Cannot find the results folder after unzipping the results.zip file!'
        report_error_message(error_message,out_path,union_key)
        logging.info('Cannot find the results folder after unzipping the results.zip file!')
        return None
    ################  
    
    try:
        gtfiles = glob.glob(os.path.join(standard_path, '*.txt'))
        #print(gtfiles)
        tsfiles = []
        for gt_txt in gtfiles:   
            ################
            #Jiang baowei 7.14 submission wrong *.txt -> log and report error
            if not os.path.exists(os.path.join(submit_dir,'results',os.path.basename(gt_txt))):
                error_message['status'] = -1
                error_message['message'] = 'Cannot find '+os.path.basename(gt_txt)+' in results/ folder!'
                report_error_message(error_message,out_path,union_key)
                logging.info('Cannot find '+os.path.basename(gt_txt)+' in results/ folder!')
                return None
            ################  
            tsfiles.append(os.path.join(submit_dir,'results',os.path.basename(gt_txt)))
        
        #print(tsfiles)
        logging.info('Found %d groundtruths and %d test files.', len(gtfiles), len(tsfiles))
        logging.info('Available LAP solvers %s', str(mm.lap.available_solvers))
        logging.info('Default LAP solver \'%s\'', mm.lap.default_solver)
        logging.info('Loading files.')
        
        gt = OrderedDict(
            [(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt, min_confidence=1)) for f in gtfiles]
        )

        ts = OrderedDict([(os.path.splitext(Path(f).parts[-1])[0], mm.io.loadtxt(f, fmt=fmt)) for f in tsfiles])

        mh = mm.metrics.create()
        accs, names, flag = compare_dataframes(gt, ts)

        # if not flag:
        #     check_code = 1
        #     report_error_msg(error_msg[check_code], 'missing txt file', out_path)
        #     return None

        metrics = list(mm.metrics.motchallenge_metrics)
        if exclude_id:
            metrics = [x for x in metrics if not x.startswith('id')]

        logging.info('Running metrics')

        summary = mh.compute_many(accs, names=names, metrics=metrics, generate_overall=True)
        print(mm.io.render_summary(summary, formatters=mh.formatters, namemap=mm.io.motchallenge_metric_names))
        logging.info('Completed')

        MOTA = summary['mota']['OVERALL']
        MOTP = 1 - summary['motp']['OVERALL']
        IDF1 = summary['idf1']['OVERALL']
        MT = summary['mostly_tracked']['OVERALL']
        ML = summary['mostly_lost']['OVERALL']
        FP = summary['num_false_positives']['OVERALL']
        FN = summary['num_misses']['OVERALL']
        IDs = summary['num_switches']['OVERALL']
        Frag = summary['num_fragmentations']['OVERALL']

        # return (MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag)

        if (MOTA + MOTP) <= 0:
            check_code = 1
            # report_error_msg(error_msg[check_code], 'MOTA + MOTP == 0', out_path)
            print("MOTA + MOTP <= 0, wrong.")
            return None
        else:
            score = (2 * MOTA * MOTP) / (MOTA + MOTP)

        score_detail = (MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag)
        score = (2 * MOTA * MOTP) / (MOTA + MOTP)

    except BaseException as e:
        print(e)
        # 写入报错信息
        # check_code = 1
        # report_error_msg(error_msg[check_code], str(e), out_path)
        if os.path.exists(os.path.join(submit_dir+'results')):
            shutil.rmtree(os.path.join(submit_dir+'results'))
        return None

    else:
        # 写入得分
        report_score(score, score_detail, out_path, union_key, status)
        if os.path.exists(os.path.join(submit_dir+'results')):
            shutil.rmtree(os.path.join(submit_dir+'results'))
        return (MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag)


def compare_dataframes(gts, ts):
    """Builds accumulator for each sequence."""
    accs = []
    names = []

    for k in gts:
        if k in ts.keys():
            logging.info('Comparing %s...', k)
            accs.append(mm.utils.compare_to_groundtruth(gts[k], ts[k], 'iou', distth=0.5))
            names.append(k)
        else:
            logging.warning('No ground truth for %s, skipping.', k)
            return accs, names, False

    return accs, names, True


def dump_2_json(info, path):
    with open(path, 'w') as output_json_file:
        json.dump(info, output_json_file, indent=4)


def report_error_msg(showMsg, detail, out_p):
    error_dict = dict()
    error_dict['errorMsg'] = showMsg
    error_dict['errorDetail'] = detail
    error_dict['score'] = 0
    error_dict['scoreJson'] = {}
    error_dict['success'] = False
    dump_2_json(error_dict, out_p)

################ 
#jiangbaowei report error message
def report_error_message(error_dict,out_p,union_key):
    error_dict['unionKey'] = union_key
    print(error_dict)
    error_dict = bytes(json.dumps(error_dict), 'utf-8')
    url = out_p
    headers={'Content-Type': 'application/json'}
    x = requests.post(url, data=error_dict,headers=headers)
################ 

def report_score(score, score_detail, out_p, union_key, status):
    if status == 0:
        msg = "Success"
    else:
        msg = "Wrong user results directory."
    url=out_p
    data={
        'unionKey': union_key,
        'Score': str(round(score, 5) if float(score) > 0.0001 else 0.0), 
        'MOTA': str(round(score_detail[0], 5) if score_detail[0] >= 0.0001 else 0.0),
        'MOTP': str(round(score_detail[1], 5) if score_detail[1] >= 0.0001 else 0.0),
        'status': status,
        'message': msg
    }
    if data['Score']=='0.0' and data['MOTA']=='0.0' and data['MOTP']=='0.0':
        data['status'] = -1
        data['message'] = 'Error!'
    #print(data)
    data = bytes(json.dumps(data), 'utf-8')
    headers={'Content-Type': 'application/json'}
    x = requests.post(url, data=data,headers=headers)

    # result = dict()
    # result['success'] = True
    # result['score'] = round(score, 5) if float(score) > 0.0001 else 0.0

    # # (MOTA, MOTP, IDF1, MT, ML, FP, FN, IDs, Frag)
    # result['scoreJson'] = {}
    # result['scoreJson']['score'] = round(score, 5) if float(score) > 0.0001 else 0.0
    # result['scoreJson']['MOTA'] = round(score_detail[0], 5) if score_detail[0] >= 0.0001 else 0.0
    # result['scoreJson']['MOTP'] = round(score_detail[1], 5) if score_detail[1] >= 0.0001 else 0.0
    # result['scoreJson']['IDF1'] = round(score_detail[2], 5) if score_detail[2] >= 0.0001 else 0.0
    # result['scoreJson']['MT'] = int(score_detail[3])
    # result['scoreJson']['ML'] = int(score_detail[4])
    # result['scoreJson']['FP'] = int(score_detail[5])
    # result['scoreJson']['FN'] = int(score_detail[6])
    # result['scoreJson']['IDs'] = int(score_detail[7])
    # result['scoreJson']['Frag'] = int(score_detail[8])

    # dump_2_json(result, out_p)


def unzip_file(zip_src, dst_dir):
    r = zipfile.is_zipfile(zip_src)
    if r:
        fz = zipfile.ZipFile(zip_src, 'r')
        for file in fz.namelist():
            fz.extract(file, dst_dir)
    else:
        print('This is not zip')


if __name__ == "__main__":
    """
    online evaluation
    """
    #if os.path.exists('./results'):
    #    shutil.rmtree('./results')
    main()
