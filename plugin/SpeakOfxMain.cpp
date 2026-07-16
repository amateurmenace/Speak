// Speak — OFX bundle entry point (standalone repo).
//
// In the original monorepo Speak was registered as a second plugin inside
// Hush's OpenNR.ofx bundle; in its own repo the bundle carries Speak alone.
// The plugin identifier stays org.opennr.Speak — it has been the identifier
// since the first commit, and changing it would orphan any saved project that
// ever used a dev build.
//
// MIT License.

#include "SpeakPlugin.h"

void OFX::Plugin::getPluginIDs(OFX::PluginFactoryArray& p_FactoryArray)
{
    speakofx::registerSpeak(p_FactoryArray);
}
