#ifndef KEYFRAMEDATABASE_H
#define KEYFRAMEDATABASE_H
#include <vector>
#include <list>
#include <set>

#include "keyframe.h"
#include "frame.h"
#include "orbvocabulary.h"

#include<mutex>

class KeyFrame;
class Frame;


class KeyFrameDatabase
{
public:

    KeyFrameDatabase(const ORBVocabulary &voc);

   void add(KeyFrame* pKF);

   void erase(KeyFrame* pKF);

   void clear();

   // Loop Detection 回环检测
   std::vector<KeyFrame *> DetectLoopCandidates(KeyFrame* pKF, float minScore);

   // Relocalization 重定位
   std::vector<KeyFrame*> DetectRelocalizationCandidates(Frame* F);

protected:

  // Associated vocabulary
  const ORBVocabulary* mpVoc; ///< 预先训练好的词典

  // Inverted file
  std::vector<std::list<KeyFrame*> > mvInvertedFile; ///< 倒排索引，mvInvertedFile[i]表示包含了第i个word id的所有关键帧

  // Mutex 互斥锁
  std::mutex mMutex;
};
#endif // KEYFRAMEDATABASE_H

