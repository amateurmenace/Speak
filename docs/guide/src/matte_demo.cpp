#include <cstdio>
#include <vector>
#include <cmath>
#include "speak_core.h"
using namespace speakcore;
static void writePPM(const char* path,const std::vector<float>& img,int W,int H){
    FILE* f=fopen(path,"wb");fprintf(f,"P6\n%d %d\n255\n",W,H);
    for(int y=H-1;y>=0;--y)for(int x=0;x<W;++x){size_t i=(size_t(y)*W+x)*4;
        for(int c=0;c<3;++c){float v=img[i+c];v=v<0?0:(v>1?1:v);fputc(int(v*255+0.5f),f);}}fclose(f);}
int main(){
    const int W=720,H=240;
    std::vector<float> src(size_t(W)*H*4),dst(src.size());
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){size_t i=(size_t(y)*W+x)*4;
        // shadow-ish mid so grain is loud (density high)
        src[i]=diEncode(0.09f);src[i+1]=diEncode(0.09f);src[i+2]=diEncode(0.09f);
        src[i+3]= (x < W/2) ? 1.0f : 0.0f;}   // hard seam: left cleaned, right protected
    SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableGrain=1;
    p.profile=neutralProfile(); p.profile.grainAmount=1.0f; p.profile.grainSize=0.25f;
    p.viewMode=SPEAK_VIEW_GRAIN; p.frameIndex=3;
    p.matteSource=SPEAK_MATTE_ALPHA; p.grainMatteFloor=0.0f;
    speakFrame(src.data(),W,H,p,dst.data()); writePPM("m_shaped.ppm",dst,W,H);
    p.matteSource=SPEAK_MATTE_OFF;
    speakFrame(src.data(),W,H,p,dst.data()); writePPM("m_uniform.ppm",dst,W,H);
    printf("ok\n");return 0;}
