# -*- coding: utf-8 -*-
"""
Created on Sun May 16 19:00:34 2021

@author: Yann
"""

import numpy as np

import scipy.signal as spsig
import scipy.optimize as spopt

import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

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
import argparse

## local
from renautils import show
from homomorphic_filtering import homomorphic_filtering

parser = argparse.ArgumentParser(description='Parse image path')
parser.add_argument('image_path')
parser.add_argument('--show', action='store_true')
parser.add_argument('--downscale', action='store_true')
parser.add_argument('--debug', action='store_true')

args = parser.parse_args()
image_path = args.image_path


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
    
def find_half_hist(img):
    hist = ske.histogram(img, nbins=256)[0]
    i, s = 0, 0
    while (s < img.shape[0]*img.shape[1]/2):
        s += hist[i]
        i+=1
    return i
    
plt.ion()
plt.close('all')
plt.rcParams['image.cmap'] = 'gray'

ticket = skio.imread(image_path, as_gray=True)
if args.downscale : ticket = skt.rescale(ticket, 1024/ticket.shape[1], preserve_range=True)

## Canny
ticket_contours = canny(ticket, sigma=2 * (ticket.shape[0]*ticket.shape[1])**.5/1500)
ticket_contours = morpho.dilation(ticket_contours, morpho.disk(3))

## Transfo de Hough
hspace, angles, distances = skt.hough_line(ticket_contours)
hspacep, angles, distances = skt.hough_line_peaks(hspace, angles, distances)
normalized_hspacep = (hspacep - np.min(hspacep)) / (np.max(hspacep) - np.min(hspacep))

## Rotation
angles_candidats = []
distances_candidats = []
for i, angle in enumerate(angles):  # filtrage des angles
    if abs(angle) < np.deg2rad(30): 
        angles_candidats.append(angle) # ils sont dans l'ordre
        distances_candidats.append(distances[i])
        
if args.debug:
    show(ticket)
    cmap = get_cmap('Wistia')
    for p, angle, dist in zip(normalized_hspacep, angles, distances):
        (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
        plt.axline((x0, y0), slope=np.tan(angle + np.pi/2), c=cmap(p))

# pour chaque angle, on détermine le multiple de pi/2 auquel il est le plus proche et son écart à ce dernier
angle = np.mean(angles_candidats[:2])
k = np.round(angle/(np.pi/2))
ecart = k*np.pi/2 - angle
phi = -ecart # angle à appliquer à l'image

ticket_rot = skt.rotate(ticket, phi*180/np.pi)

## Recadrage bords verticaux
center = (ticket.shape[1] / 2 - 0.5, ticket.shape[0] / 2 - 0.5)
Xbords = []
for angle, dist in zip(angles_candidats[:2], distances_candidats[:2]):
    (x0, y0) = dist * np.array([np.cos(angle), np.sin(angle)])
    (x1, y1) = rotate((x0,y0), origin=center, degrees=-phi*180/np.pi)
    Xbords.append(x1)
Xbords = np.round(sorted(Xbords)).astype(int)
if len(Xbords)>=2:
    ticket_rot_crop = ticket_rot[:,Xbords[0]:Xbords[1]]
else:
    print("WARNING: couldn't crop")
    ticket_rot_crop = ticket_rot

if args.debug : show(ticket, ticket_contours, ticket_rot, ticket_rot_crop, nrows=1)

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
    profil[u] = np.sum(ticket2_contours[a:b])
    
peaks, _ = spsig.find_peaks(profil, height=0.6*np.max(profil), prominence=0.4*np.mean(profil), distance=7)

if args.debug:
    plt.figure()
    plt.plot(profil)
    plt.scatter(peaks, profil[peaks])
    plt.show()

# on prendre le deuxième pic en partant de la gauche, et le deuxième en partant de la droite
peaks = [x for x in peaks if HAUTEUR_BLOC<=x<=len(profil)-HAUTEUR_BLOC]
if args.debug : print("Peaks:\n",peaks)

#### au cas où on n'aurait pas comme prévu un pic au début et un pic à la fin (par ex quand la photo coupe le ticket)
a = 0
b = ticket2.shape[0]
if len(peaks)>0:
    if peaks[0] < ticket2.shape[0]*1/3:
        a = peaks[0]
    if peaks[-1] > ticket2.shape[0]*2/3:
        b = peaks[-1]
####

ticket2_crop = ticket2[a:b, :]

if args.debug : show(ticket2_crop, 'fin crop')

### OCR

## filtrage homomorphique
ticket_avant_filtrage = img_as_float(ticket2_crop, True)
#ticket_avant_filtrage[ticket2_crop < np.quantile(ticket2_crop,0.05)] = find_half_hist(ticket2_crop)/255
fc = 60 * (ticket2_crop.shape[1]/1500) / ticket2_crop.shape[0]
ticket_ocr_hom = homomorphic_filtering(ticket_avant_filtrage, fc=fc)

if args.debug : show(ticket_ocr_hom, title='filtrage homomorphique')

ticket_ocr = img_as_ubyte(ticket_ocr_hom)

if args.debug: show(ticket2_crop, ticket_ocr, title="avant début optimisation")

def levels(img, a, b, gamma):
    assert 0<=a<b<=1
    img = img_as_float(img, True)
    img = np.clip(img, a, b)
    img = (img-a)/(b-a) # niveau
    img = ske.adjust_gamma(img, gamma)
    return img_as_ubyte(img)

def _OCR_score(img, a, b, gamma, sigma, order='blur-first', expected_words=['total']):
    ''' order: 'blur-first', 'levels-first' '''
    assert ticket_ocr.dtype == np.uint8
    assert 0<=a<b<=1
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
res = spopt.minimize_scalar(lambda x: -_OCR_score(ticket_ocr_hom, 0, half_hist/255, x, 0, order='levels-first', expected_words=['total','auchan','supermarche', 'stalingrad']), bounds=(0.01,9.99), method='bounded', options={'disp':3, 'xatol':0.1})
gamma, conf = res.x, -res.fun

best_sigma, best_conf = 0, 0
for sigma in [0.2, 0.3, 0.4, 0.5]:
    print("finding best sigma, currently: sigma=",sigma,sep="",end="")
    conf = _OCR_score(ticket_ocr_hom, 0, half_hist/255, gamma, sigma, order='levels-first', expected_words=['total','auchan','supermarche', 'stalingrad'])
    print("\tconf=",conf,sep="")
    if conf < best_conf : break
    elif conf == best_conf: continue
    elif conf > best_conf : best_sigma, best_conf = sigma, conf

best_gamma = gamma

ticket_ocr = levels(ticket_ocr_hom, 0, half_hist/255, best_gamma)
ticket_ocr = img_as_ubyte(skf.gaussian(img_as_float(ticket_ocr), sigma=best_sigma)) 
    
ocr_data = pytesseract.image_to_data(ticket_ocr, output_type=pytesseract.Output.DICT)

if args.show == True:
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(ticket_ocr)
    plt.subplot(1,2,2)
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
        if c in ['l','i','I']:
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

if args.show == True:
    plt.show()