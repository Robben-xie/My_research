#ifndef ORBVOCABULARY_H
#define ORBVOCABULARY_H
#include <opencv2/core/core.hpp>

#include "./FORB.h"
#include "./TemplatedVocabulary.h"

typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB>
  ORBVocabulary;

#endif // ORBVOCABULARY_H
