#!/bin/csh
gmt set FONT_TITLE             14p,Helvetica-Bold,black
gmt set FONT_LABEL             11p,Helvetica-Bold,black
gmt set FONT_ANNOT_PRIMARY     11p,Helvetica-Bold,black
gmt set MAP_GRID_PEN_PRIMARY    0.08p,black
gmt set MAP_GRID_PEN_SECONDARY  0.05p,black

set INPUT=mlp
set  PS=${INPUT}.lune.ps
set JPG=${INPUT}.lune.jpg
set R=" -R-30/30/-90/90 "
set J=" -JH0/2i "

#### create surface grid from lon, lat, pVR points  -Vl(long verbosy) -Vq(quiet) -Vn(err only)
####
# 1        2        3    4    5    6     7          8       9       10
# lune_lat,lune_lon,eig1,eig2,eig3,class,etype_pred,prob_eq,prob_ex,prob_co
# -17.165,-3.001,-0.9528,-0.2441,0.6063,0,eq,0.9993766647175392,8.481511468654373e-05,0.0005385201677744537
#
# awk -F, '{print $2, $1, $8}' ${INPUT}.csv > ${INPUT}.eq
# awk -F, '{print $2, $1, $9}' ${INPUT}.csv > ${INPUT}.ex
# awk -F, '{print $2, $1, $10}' ${INPUT}.csv > ${INPUT}.co
# awk -F, '{print $2, $1, $6}' ${INPUT}.csv > ${INPUT}.class

# 1        2        3    4    5    6   7   8   9   10  11  12    13         14      15      16
# lune_lat,lune_lon,eig1,eig2,eig3,mxx,myy,mzz,mxy,mxz,myz,class,etype_pred,prob_eq,prob_ex,prob_co
# -17.165,-3.001,-0.9528,-0.2441,0.6063,-0.1772,-0.499,0.08557,-0.08883,0.2217,-0.6831,0,eq,0.9963317942559868,0.0015529424476224127,0.002115263296390734
#
# awk -F, '{print $2, $1, $14}' ${INPUT}.csv > ${INPUT}.eq
# awk -F, '{print $2, $1, $15}' ${INPUT}.csv > ${INPUT}.ex
# awk -F, '{print $2, $1, $16}' ${INPUT}.csv > ${INPUT}.co
# awk -F, '{print $2, $1, $12}' ${INPUT}.csv > ${INPUT}.class

# lune_lat,lune_lon,class,etype_pred,prob_eq,prob_ex,prob_co
# -17.165,-3.001,0,eq,0.9895550773643719,0.00042012585402395256,0.01002479678160413
#
awk -F, '{print $2, $1, $5}' ${INPUT}.csv > ${INPUT}.eq
awk -F, '{print $2, $1, $6}' ${INPUT}.csv > ${INPUT}.ex
awk -F, '{print $2, $1, $7}' ${INPUT}.csv > ${INPUT}.co
awk -F, '{print $2, $1, $3}' ${INPUT}.csv > ${INPUT}.class

gmt surface ${INPUT}.eq    $R -I1/1 -G${INPUT}.eq.grd    -T1 -C1.0e-4 -Ll0 -Lu1 -Vn
gmt surface ${INPUT}.ex    $R -I1/1 -G${INPUT}.ex.grd    -T1 -C1.0e-4 -Ll0 -Lu1 -Vn
gmt surface ${INPUT}.co    $R -I1/1 -G${INPUT}.co.grd    -T1 -C1.0e-4 -Ll0 -Lu1 -Vn
gmt surface ${INPUT}.class $R -I1/1 -G${INPUT}.class.grd -T1 -C1.0e-4 -Ll-1 -Lu3 -Vn

### create the points and label for the special Lune DC,+/-ISO,+/-CLVD,+/-LVD,+/-Crack 
###
### format: lon lat label justify-code 
###
cat >! points.xy << EOF
  0   0  DC    0 
  0 -90 -ISO   1 
  0 +90 +ISO   2 
+30   0 -CLVD  1 
+30 -60 -Crack 1 
+30 -35 -LVD   1 
-30 +60 +Crack 2 
-30 +35 +LVD   2 
-30   0 +CLVD  2 
EOF

#### Contour interval
####
cat >! cint.xy << EOF
0.5 C
0.9
EOF

### Color Palete   ####
#### -Z continous 24bit color ###
gmt makecpt -Crainbow -T0/1/0.01 -D -V >! prob.cpt
gmt makecpt -Ccategorical -T-1/3/1 -V >! class.cpt

###########################################################
#### Start GMT plot - plot the grid image
####
# gmt grdimage $R $J ${INPUT}.class.grd -Cclass.cpt -K -Vq >! ${PS}

gmt psbasemap $R $J -Bxf180g10a180 -Byf180g10a180 -Bnsew+t"${INPUT} classes" -K -Vq >! ${PS}

#### plot the contours on the lune
####
# gmt grdcontour ${INPUT}.eq.grd $R $J -Ccint.xy -W1p,black -O -K -Vn  -L+-1/+3 -S100 -Gd5i -A+20+ap+c0.02i+o+f12p,Helvetica-Bold,black+gwhite+p0.25p,black >> ${PS}

# gmt psxy ${INPUT}.class $R $J -Sc0.03i -Cclass.cpt -W0.25p,black -O -K -Vq >> ${PS}
gmt psxy ${INPUT}.class $R $J -Sc0.03i -Cclass.cpt -O -K -Vq >> ${PS}

### plot the special lune points and labels DC,+/-ISO,+/-CLVD,+/-LVD,+/-Crack
####
gmt psxy points.xy $R $J -N -Sc0.05i -W0.5p,black -Gblack -O -K -Vq >> ${PS}

### DC only
awk '{ if( $4 == 0 ) print $1, $2, $3 }' points.xy | pstext $R $J -N -D+0.12i/+0.12i -F+f12p,Helvetica-Bold,black+jBL -C0.01i -Gwhite -W0.1p,black -O -K -Vq >> ${PS}
### CLVD, LVD, ISO
awk '{ if( $4 == 2 ) print $1, $2, $3 }' points.xy | pstext $R $J -N -D-0.12i/0.0i   -F+f12p,Helvetica-Bold,black+jMR -C0.01i -Gwhite -W0.1p,black -O -K -Vq >> ${PS}
awk '{ if( $4 == 1 ) print $1, $2, $3 }' points.xy | pstext $R $J -N -D+0.12i/0.0i   -F+f12p,Helvetica-Bold,black+jML -C0.01i -Gwhite -W0.1p,black -O -K -Vq >> ${PS}

awk '{print $1,$2}' GMT_/sourcetype_arc_03.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_04.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_05.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_06.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}

###########################################################
#### Start GMT plot - plot the grid image
####
gmt grdimage $R $J ${INPUT}.eq.grd -Cprob.cpt -K -O -Vq -X2.75i >> ${PS}
gmt psbasemap $R $J -Bxf180g10a180 -Byf180g10a180 -Bnsew+t"${INPUT} probs eq" -O -K -Vq >> ${PS}

#### plot the contours on the lune
####
# gmt grdcontour ${INPUT}.eq.grd $R $J -Ccint.xy -W1p,black -O -K -Vn  -L+0/+1 -S100 -Gd5i -A+20+ap+c0.02i+o+f12p,Helvetica-Bold,black+gwhite+p0.25p,black >> ${PS}

### plot the special lune points and labels DC,+/-ISO,+/-CLVD,+/-LVD,+/-Crack 
gmt psxy points.xy $R $J -N -Sc0.05i -W0.5p,black -Gblack -O -K -Vq >> ${PS}

awk '{print $1,$2}' GMT_/sourcetype_arc_03.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_04.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_05.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_06.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}

#### Color scale
####
# gmt psscale -Dx2.5i/0.5i+w1i/0.1i+e0.1i -Bf0.1a0.2g0.2+l"Probability" -Cprob.cpt  -O -K -Vq >> ${PS}

###########################################################
#### Start GMT plot - plot the grid image
####
gmt grdimage $R $J ${INPUT}.ex.grd -Cprob.cpt -K -O -Vq -X2.25i >> ${PS}
gmt psbasemap $R $J -Bxf180g10a180 -Byf180g10a180 -Bnsew+t"${INPUT} probs ex" -O -K -Vq >> ${PS}

#### plot the contours on the lune
####
# gmt grdcontour ${INPUT}.ex.grd $R $J -Ccint.xy -W1p,black -O -K -Vn  -L+0/+1 -S100 -Gd5i -A+20+ap+c0.02i+o+f12p,Helvetica-Bold,black+gwhite+p0.25p,black >> ${PS}

### plot the special lune points and labels DC,+/-ISO,+/-CLVD,+/-LVD,+/-Crack
####
gmt psxy points.xy $R $J -N -Sc0.05i -W0.5p,black -Gblack -O -K -Vq >> ${PS}

awk '{print $1,$2}' GMT_/sourcetype_arc_03.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_04.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_05.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_06.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}

#### Color scale
####
# gmt psscale -Dx2.5i/0.5i+w1i/0.2i+e0.1i -Bf0.1a0.2g0.2+l"Probability" -Cprob.cpt  -O -K -Vq >> ${PS}

###########################################################
#### Start GMT plot - plot the grid image
####
gmt grdimage $R $J ${INPUT}.co.grd -Cprob.cpt -K -O -Vq -X2.25i >> ${PS}
gmt psbasemap $R $J -Bxf180g10a180 -Byf180g10a180 -Bnsew+t"${INPUT} probs co" -O -K -Vq >> ${PS}

#### plot the contours on the lune
####
# gmt grdcontour ${INPUT}.co.grd $R $J -Ccint.xy -W1p,black -O -K -Vn  -L+0/+1 -S100 -Gd5i -A+20+ap+c0.02i+o+f12p,Helvetica-Bold,black+gwhite+p0.25p,black >> ${PS}

### plot the special lune points and labels DC,+/-ISO,+/-CLVD,+/-LVD,+/-Crack
####
gmt psxy points.xy $R $J -N -Sc0.05i -W0.5p,black -Gblack -O -K -Vq >> ${PS}

awk '{print $1,$2}' GMT_/sourcetype_arc_03.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_04.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_05.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}
awk '{print $1,$2}' GMT_/sourcetype_arc_06.dat | gmt psxy $R $J -W1p,black,5_2:0p -O -K >> ${PS}

#### Color scale
####
gmt psscale -Dx2i/0.5i+w1i/0.2i+e0.1i -Bf0.1a0.2g0.2+l"Probability" -Cprob.cpt  -O -Vq >> ${PS}

#### convert and deep clean
####
psconvert -Tj -E600 -A ${PS}

/bin/rm -f ${PS}
/bin/rm -f prob.cpt points.xy cint.xy

open ${JPG}
