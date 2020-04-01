def compute_f1(Y_pred, Y):
  other_class = 4
  tp = 0
  fp = 0
  fn = 0
  for sample_indx, sample in enumerate(Y):
    for predict_indx, each_predict in enumerate(sample):
      if(sample[predict_indx]==-100):
        break
      if(sample[predict_indx]!=other_class and Y_pred[sample_indx][predict_indx]!=other_class):
        tp += 1
      elif(sample[predict_indx]!=other_class and Y_pred[sample_indx][predict_indx]==other_class):
        fp += 1
      elif(sample[predict_indx]==other_class and Y_pred[sample_indx][predict_indx]!=other_class):
        fn += 1
  precision = tp/(tp+fp)
  recall = tp/(tp+fn)
  return [precision, recall, 2*(precision*recall)/(precision+recall)]
