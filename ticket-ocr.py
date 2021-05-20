# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:00:34 2021

@author: Yann
"""

import numpy as np

import scipy.signal as spsig
import scipy.optimize as spopt

import matplotlib.pyplot as plt

import skimage.io as skio
import skimage.filters as skf
import skimage.transform as skt
from skimage import morphology as morpho
import skimage.exposure as ske
from skimage.util import img_as_ubyte, img_as_float
from skimage.feature import canny

import pytesseract

from math import floor, ceil
import json
import sys

## local
from renautils import show
from homofiltrer import homofiltrer


def rotate(p, origin=(0, 0), degrees=0):
    ''' returns the rotated point p around origin '''
    angle = np.deg2rad(degrees)
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle),  np.cos(angle)]])
    o = np.atleast_2d(origin)
    p = np.atleast_2d(p)
    return np.squeeze((R @ (p.T-o.T) + o.T).T)

def plotOCR(ocr_data):
    ''' plot ocr text in current figure '''
    n = len(ocr_data['text'])
    for i in range(n):
        text = ocr_data['text'][i]
        row = ocr_data['top'][i]
        col = ocr_data['left'][i]
        #print(row, col, text)
        plt.text(col, ticket_ocr.shape[0]-row-1, text)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim((0, ticket_ocr.shape[1]))
    plt.ylim((0, ticket_ocr.shape[0]))
    

plt.close('all')
plt.rcParams['image.cmap'] = 'gray'

ticket = skio.imread('ticket5.jpg', as_gray=True)
#if ticket.shape[1]>1024 : ticket = skt.rescale(ticket, 1024/ticket.shape[1], preserve_range=True)

## Canny
ticket_contours = canny(ticket)

## Transfo de Hough
hspace, angles, distances = skt.hough_line(ticket_contours)
hspacep, angles, distances = skt.hough_line_peaks(hspace, angles, distances)

## Rotation
# on ne prend que le premier pic qui correspond à un bord vertical du ticket
# pour chaque angle, on détermine le multiple de pi/2 auquel il est le plus proche et son écart à ce dernier
k = np.round(angles[0]/(np.pi/2))
ecart = k*np.pi/2 - angles[0]
phi = -ecart # angle à appliquer à l'image

ticket_rot = skt.rotate(ticket, phi*180/np.pi)

## Recadrage bords verticaux
center = (ticket.shape[1] / 2 - 0.5, ticket.shape[0] / 2 - 0.5)
Xbords = []
for angle, dist in zip(angles[:2], distances[:2]):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    (x1, y1) = rotate((x0,y0), origin=center, degrees=-phi*180/np.pi)
    Xbords.append(x1)
Xbords = np.round(sorted(Xbords)).astype(int)
ticket_rot_crop = ticket_rot[:,Xbords[0]:Xbords[1]]

# show(ticket, ticket_contours, ticket_rot, ticket_rot_crop, nrows=1)

## Recadrage bords horizontaux
ticket2 = ticket_rot_crop[:,:]
ker = morpho.rectangle(12,1)
ticket2_contours = morpho.dilation(ticket_rot_crop, ker) - morpho.erosion(ticket_rot_crop, ker)

# profil luminosité vertical bloc
HAUTEUR_BLOC = int(8 * ticket2.shape[1]/1024) # ticket2.shape[1]/1024 est un coeff d'échelle
profil = np.zeros(ticket2.shape[0])
for u in range(ticket2.shape[0]):
    a = max(u - floor(HAUTEUR_BLOC/2), 0)
    b = min(u + ceil(HAUTEUR_BLOC/2), ticket2.shape[0])
    profil[u] = np.sum(ticket2_contours[u-HAUTEUR_BLOC//2:u+HAUTEUR_BLOC//2])
    
peaks, _ = spsig.find_peaks(profil, height=0.6*np.max(profil), prominence=0.4*np.mean(profil), distance=7)

# on prendre le deuxième pic en partant de la gauche, et le deuxième en partant de la droite
peaks = [x for x in peaks if 2*HAUTEUR_BLOC<=x<=len(profil)-2*HAUTEUR_BLOC]
ticket2_crop = ticket2[peaks[0]:peaks[-1], :]


### OCR
ticket_ocr = img_as_ubyte(ticket2_crop)

ticket_ocr_hom = homofiltrer(ticket_ocr, fc=0.02)
ticket_ocr_hom = img_as_ubyte(ticket_ocr_hom)

ticket_ocr = ticket_ocr_hom

def levels(img, a, b, gamma):
    img = np.copy(img)
    img[img > b] = b
    img[img < a] = a
    img = img_as_float(img)
    af, bf = a/255, b/255
    img = (img-af)/(bf-af) # niveau
    img = ske.adjust_gamma(img, gamma)
    return img_as_ubyte(img)

def _OCR_score(img, a, b, gamma, sigma, order='blur-first', expected_words=['total']):
    ''' order: 'blur-first', 'levels-first' '''
    assert ticket_ocr.dtype == np.uint8
    assert 0<=a<b<=255
    assert sigma>=0
    if order=='blur-first':
        img = img_as_ubyte(skf.gaussian(img_as_float(img), sigma=sigma)) 
        img = levels(img, a, b, gamma)
    elif order=='levels-first':
        img = levels(img, a, b, gamma)
        img = img_as_ubyte(skf.gaussian(img_as_float(img), sigma=sigma)) 
    else:
        raise BaseException
    ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
    
    # ancien calcul de conf
    # conf = sum([float(x) for x in ocr_data['conf']])/len(ocr_data['conf'])
    

    score = 0
    for i in range(len(ocr_data['text'])):
        row, col, text = ocr_data['top'][i], ocr_data['left'][i], ocr_data['text'][i]
        for word in expected_words : 
            if text.upper().find(word.upper()) >= 0 : score+=2
        if col > 0.68*img.shape[1] and text != "":
            for c in text:
                if c in ['0','1','2','3','4','5','6','7','8','9','.',',','€']:
                    score += 1
                    if c=='€': score+=2
                else:
                    score -= 1
    
    
    return score

def find_half_hist(img):
    hist = ske.histogram(img)[0]
    i, s = 0, 0
    while (s < img.shape[0]*img.shape[1]/2):
        s += hist[i]
        i+=1
    return i



half_hist = find_half_hist(ticket_ocr_hom)


# ## opti sigma
# max_conf, max_conf_sigma, max_conf_gamma = 0, 0, 0
# for sigma in [0,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]:
#     print("finding best sigma and gamma, currently: sigma=",sigma,sep="",end="")
#     res = spopt.minimize_scalar(lambda x: -_OCR_score(ticket_ocr_hom, 0, half_hist, x, sigma, order='levels-first', expected_words=['total','auchan','supermarche', 'stalingrad']), bounds=(0.01,9.99), method='bounded', options={'disp':1, 'xatol':0.2})
#     gamma, conf = res.x, -res.fun
#     print("\tgamma=",gamma,"\tconf=",conf,sep="")
#     if conf >= max_conf: max_conf, max_conf_sigma, max_conf_gamma = conf, sigma, gamma
#     else: break
# print(max_conf)

## opti gamma puis sigma
print("finding best gamma")
res = spopt.minimize_scalar(lambda x: -_OCR_score(ticket_ocr_hom, 0, half_hist, x, 0, order='levels-first', expected_words=['total','auchan','supermarche', 'stalingrad']), bounds=(0.01,9.99), method='bounded', options={'disp':3, 'xatol':0.1})
gamma, conf = res.x, -res.fun

best_sigma, best_conf = 0, 0
for sigma in [0.2, 0.3, 0.4, 0.5]:
    print("finding best sigma, currently: sigma=",sigma,sep="",end="")
    conf = _OCR_score(ticket_ocr_hom, 0, half_hist, gamma, sigma, order='levels-first', expected_words=['total','auchan','supermarche', 'stalingrad'])
    print("\tconf=",conf,sep="")
    if conf < best_conf : break
    elif conf == best_conf: continue
    elif conf > best_conf : best_sigma, best_conf = sigma, conf

best_gamma = gamma

ticket_ocr = levels(ticket_ocr_hom, 0, half_hist, best_gamma)
ticket_ocr = img_as_ubyte(skf.gaussian(img_as_float(ticket_ocr), sigma=best_sigma)) 
    
ocr_data = pytesseract.image_to_data(ticket_ocr, output_type=pytesseract.Output.DICT)

plt.figure()
plotOCR(ocr_data)

def sanitizePrice(text):
    # version gentille pour le moment
    # on cherche les chiffres, on coupe là où y a le séparateur
    nbrs_str = ""
    sep_i = -1
    for i, c in enumerate(text):
        if sep_i>=0 and len(nbrs_str)-sep_i==2:   # si on a déjà deux chiffres après la virgule
            break
        if c in ['0','1','2','3','4','5','6','7','8','9']:
            nbrs_str += c
        if c in ('o','O','g'):
            nbrs_str += '0'
        if c in ['l']:
            nbrs_str += '1'
        if c in ('S'):
            nbrs_str += '5'
        if c in ('B'):
            nbrs_str += '8'
        if c in ['.', ',']:
            sep_i = len(nbrs_str) # sep_i est l'indice de l'élément de nbrs_str avant lequel placer la virgule
    if sep_i==-1: return None
    else :
        price_str = nbrs_str[:sep_i] + '.' + nbrs_str[sep_i:]
        try: return float(price_str)
        except: return None
        
# def determine_line_height(ocr_data, image_width):
#     rows = []
#     for i in range(len(ocr_data['text'])):
#         row, col, text = ocr_data['top'][i], ocr_data['left'][i], ocr_data['text'][i]
#         if 0.68 < col/image_width < 0.95  and ('.' in text or ',' in text):
#             rows.append(row)
#     heights = []
#     for i in range(1,len(rows)):
#         heights.append(rows[i]-rows[i-1])
#     return np.quantile(heights, 0.2)

if True:
    ticket_dict = {}
    ticket_dict['score'] = best_conf
    ticket_dict['articles'] = []
    
    row_total = ticket_ocr.shape[0]
    for i in range(len(ocr_data['text'])):
        if ocr_data['text'][i].upper().find("TOTAL") >= 0: 
            row_total = ocr_data['top'][i]
            break
        
            
    
    for i in range(len(ocr_data['text'])):
        
        row, col, text, width, height = ocr_data['top'][i], ocr_data['left'][i], ocr_data['text'][i], ocr_data['width'][i], ocr_data['height'][i]
        
        if text not in ("", " ") :
            
            if 0.68 < col/ticket_ocr.shape[1] < 0.95 and row < row_total + height/2:
                
                if abs(row-row_total)>=height/2 :  # si c'est un article
                    nomArticle = ""
                    
                    for j in range(len(ocr_data['text'])):
                        
                        row_, col_, text_ = ocr_data['top'][j], ocr_data['left'][j], ocr_data['text'][j]
                        
                        if col_ < 0.68*ticket_ocr.shape[1] :
                            if abs(row_-row) < height/2 and text_ not in ("", " "):
                                nomArticle += text_ + " "
                                
                    ticket_dict['articles'].append({'name': nomArticle,
                                                    'price_str': text,
                                                    'price': sanitizePrice(text)
                                                   })
                else :
                    ticket_dict['total'] = sanitizePrice(text)
                        
    
    #print(ticket_dict)
    
    print(json.dumps(ticket_dict, sort_keys=True, indent=4))
