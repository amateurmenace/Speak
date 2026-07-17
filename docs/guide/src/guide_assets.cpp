#include <cstdio>
#include <vector>
#include <cmath>
#include <string>
#include "speak_core.h"
using namespace speakcore;

static void writePPM(const std::string& path, const std::vector<float>& img, int W, int H){
    FILE* f=fopen(path.c_str(),"wb");fprintf(f,"P6\n%d %d\n255\n",W,H);
    for(int y=H-1;y>=0;--y)for(int x=0;x<W;++x){size_t i=(size_t(y)*W+x)*4;
        for(int c=0;c<3;++c){float v=img[i+c];v=v<0?0:(v>1?1:v);fputc(int(v*255+0.5f),f);}}fclose(f);}

// A synthetic but photographic-feeling test frame: graded sky gradient, a
// bright practical (halation/bloom bait), a mid skin patch, shadow texture.
static void scene(std::vector<float>& src,int W,int H,bool alphaMatte){
    src.assign(size_t(W)*H*4,0.0f);
    for(int y=0;y<H;++y)for(int x=0;x<W;++x){
        size_t i=(size_t(y)*W+x)*4;
        float u=float(x)/(W-1), v=float(y)/(H-1);
        // sky: warm gradient top to bottom
        float sky = 0.10f + 0.55f*(1.0f-v);
        float r=sky*1.05f, g=sky*0.98f, b=sky*0.90f;
        // shadow band bottom third with fine texture
        if(v<0.34f){ float sh=0.02f+0.05f*v; float n=0.5f+0.5f*std::sin(x*0.9f)*std::cos(y*0.7f);
            r=g=b=sh*(0.7f+0.6f*n); r*=1.02f; }
        // mid skin patch
        float dx=u-0.30f, dy=v-0.55f;
        if(dx*dx*2.2f+dy*dy<0.02f){ r=0.34f; g=0.24f; b=0.18f; }
        // bright practical (small hot disc)
        float ex=u-0.72f, ey=v-0.62f; float rr=std::sqrt(ex*ex+ey*ey);
        if(rr<0.05f){ float k=1.0f-rr/0.05f; r=g=b=0.9f+6.0f*k*k; r*=1.0f; g*=0.96f; b*=0.9f; }
        src[i]=diEncode(r);src[i+1]=diEncode(g);src[i+2]=diEncode(b);
        // matte: left half cleaned (1), right half protected motion (0), soft seam
        src[i+3]= alphaMatte ? clampf((0.62f-u)*6.0f,0.0f,1.0f) : 1.0f;
    }
}

int main(){
    const int W=720,H=405;
    std::vector<float> src, dst(size_t(W)*H*4);
    SpeakProfile look = stockProfile(SPEAK_STOCK_PUNCH);
    look.halAmount=0.7f; look.halRadius=1.2f; look.halThresh=0.6f;
    look.bloomAmount=0.35f; look.bloomRadius=4.0f; look.bloomVeil=0.1f;
    look.grainAmount=0.55f; look.grainSize=0.14f;
    // 1. input (identity)
    scene(src,W,H,false);
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.profile=neutralProfile();
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_input.ppm",dst,W,H); }
    // 2. full look
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableTone=1;p.strength=1.0f;
      p.enableDye=1;p.enableOptics=1;p.enableGrain=1; p.profile=look;
      for(int c=0;c<3;++c){p.profile.subSat[c]=0.5f;p.profile.subSatKnee[c]=2.2f;}
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_look.ppm",dst,W,H); }
    // 3. grain isolated (on gray)
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableGrain=1;
      p.profile=look; p.viewMode=SPEAK_VIEW_GRAIN; p.frameIndex=7;
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_grain.ppm",dst,W,H); }
    // 4a. matte-shaped grain (grain isolated, alpha matte on) — left full, right floor
    scene(src,W,H,true);
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableGrain=1;
      p.profile=look; p.viewMode=SPEAK_VIEW_GRAIN; p.frameIndex=7;
      p.matteSource=SPEAK_MATTE_ALPHA; p.grainMatteFloor=0.2f;
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_matte_shaped.ppm",dst,W,H); }
    // 4b. uniform grain (no matte) for comparison
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableGrain=1;
      p.profile=look; p.viewMode=SPEAK_VIEW_GRAIN; p.frameIndex=7;
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_matte_uniform.ppm",dst,W,H); }
    // 5. halation isolated
    scene(src,W,H,false);
    { SpeakParams p={}; p.inputColorSpace=SPEAK_CS_DWG_INTERMEDIATE; p.enableTone=1;p.strength=1.0f;
      p.enableOptics=1; p.profile=look; p.viewMode=SPEAK_VIEW_SCATTER;
      speakFrame(src.data(),W,H,p,dst.data()); writePPM("g_halation.ppm",dst,W,H); }
    printf("wrote guide assets\n");return 0;
}
