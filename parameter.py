# -*- coding: utf-8 -*-
"""
Created on Thu Mar  4 12:17:37 2021

@author: fangyan
"""
import argparse

def get_parser():
    parser = argparse.ArgumentParser(description = 'parameters')
    parser.add_argument('--WHITE',default='#FFFFFF',type=str,help='Label color')
    parser.add_argument('--GREY',default='#999999',type=str,help='Label color')
    parser.add_argument('--RED',default='#E41A1C',type=str,help='Label color')                    
    parser.add_argument('--PURPLE',default='#984EA3',type=str,help='Label color')
    parser.add_argument('--BROWN',default='#A65628',type=str,help='Label color') 
    parser.add_argument('--LIGHT_BROWN',default='#C55A11',type=str,help='Label color')
    parser.add_argument('--LIGHT_PURPLE',default='#C4A8A8',type=str,help='Label color')
    parser.add_argument('--label',default='[1,2,3,4,5]',type=list,help='Label list')
    parser.add_argument('--SEGMENTS',default='4',type=int,help='Number of segments')
    parser.add_argument('--SUBJECT',default='intra',type=str,help='intra- or inter- subject')
    parser.add_argument('--regisMethod',default='Affine',type=str,help='Affine or Rigid or SyN')
    
    args = parser.parse_args()
    if args.SUBJECT == 'intra':
        parser.add_argument('--SLIDEAXIAL',default='122',type=int,help='slide number of axial plane')
        parser.add_argument('--SLIDE',default='122',type=int,help='slide number of coronal plane')
        parser.add_argument('--waxholmLabeltoMRPath',default='../data/intrasubject/07waxholmLabel_to_FAMRI_elastixBspline.nii.gz',type=str,help='path of waxholm_label_to_MR')
        parser.add_argument('--waxholmLabeltoLSPath',default='../data/intrasubject/waxholmLabel_to_LSFSL.nii.gz',type=str,help='path of waxholm_label_to_LS')
        parser.add_argument('--mrpath',default='../data/intrasubject/02raw_FAMRI_brainBET.nii.gz',type=str,help='path of FA-MR')
        parser.add_argument('--lspath',default='../data/intrasubject/01LS_to_FAMRI_fslAffine.nii.gz',type=str,help='path of LS')
    elif args.SUBJECT == 'inter':
        parser.add_argument('--SLIDEAXIAL',default='85',type=int,help='slide number of axial plane')
        parser.add_argument('--SLIDE',default='113',type=int,help='slide number of coronal plane')
        parser.add_argument('--waxholmLabeltoMRPath',default='../data/intersubject/waxholmLabeltoMRIFA_las.nii.gz',type=str,help='path of waxholm_label_to_MR')
        parser.add_argument('--waxholmLabeltoLSPath',default='../data/intersubject/waxholmLabeltoLS_las.nii.gz',type=str,help='path of waxholm_label_to_LS')
        parser.add_argument('--mrpath',default='../data/intersubject/FaSkullStripping_las.nii.gz',type=str,help='path of FA-MR')
        parser.add_argument('--lspath',default='../data/intersubject/lsAntsAffine_las.nii.gz',type=str,help='path of LS')
                   
    return parser
