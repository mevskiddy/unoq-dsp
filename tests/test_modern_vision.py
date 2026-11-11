import unittest
import numpy as np
from unoq_dsp.modern import YOLODecoder,prepare_profile,profile_lookup

class ModernVisionTests(unittest.TestCase):
 def test_lookup_matches_float32_export_for_every_byte(self):
  for mean,std,scale,zero in [((.485,.456,.406),(.229,.224,.225),.018249010667,-13),((0,0,0),(1,1,1),1/255,-128)]:
   pixels=np.repeat(np.arange(256,dtype='u1')[:,None],3,axis=1)
   expected=np.clip(np.rint(((pixels.astype('f4')/np.float32(255)-np.asarray(mean,'f4'))/np.asarray(std,'f4'))/np.float32(scale))+zero,-128,127).astype('i1')
   np.testing.assert_array_equal(profile_lookup(mean,std,scale,zero,'|i1'),expected)
 def detector(self):
  names=[f'out{i}' for i in range(6)];specs=[];outputs={}
  for i,stride in enumerate((8,16,32)):
   h=64//stride
   for j,c in ((0,4),(1,2)):
    name=names[2*i+j];shape=(1,h,h,c);specs.append({'name':name,'shape':shape,'dtype':'int8','quantization':{'scale':.125 if j==0 else .25,'zero_point':0}})
    outputs[name]=np.full(shape,8 if j==0 else -100,'i1')
  task={'kind':'yolo_one_to_one','schema':1,'labels':['a','b'],'num_classes':2,'strides':[8,16,32],'output_order':names}
  return YOLODecoder(task,specs,(1,64,64,3)),outputs
 def test_decode_grid_logits_and_inverse_letterbox(self):
  decoder,outputs=self.detector();outputs['out1'][0,2,3,1]=8
  result=decoder.decode(outputs,geometry={'ratio':.5,'left':0,'top':16,'width':128,'height':64})
  self.assertEqual(len(result),1);self.assertEqual(result[0].label,'b');self.assertAlmostEqual(result[0].score,1/(1+np.exp(-2)),6)
  np.testing.assert_allclose(result[0].box,(.3125,0,.5625,.375))
 def test_multiple_classes_can_share_an_anchor_without_nms(self):
  decoder,outputs=self.detector();outputs['out1'][0,2,3]=[8,7]
  result=decoder.decode(outputs,geometry={'ratio':1,'left':0,'top':0,'width':64,'height':64})
  self.assertEqual([r.class_id for r in result],[0,1]);self.assertEqual(result[0].box,result[1].box)
 def test_invalid_confidence_is_rejected(self):
  decoder,outputs=self.detector()
  with self.assertRaises(ValueError):decoder.decode(outputs,geometry={},score_threshold=1.2)
 def test_classifier_crop_preserves_constant_color(self):
  from PIL import Image
  spec={'shape':(1,224,224,3),'dtype':'int8','quantization':{'scale':.01,'zero_point':-128}}
  image=Image.new('RGB',(341,199),(20,80,150));profile={'resize':'center_crop','crop_pct':.875,'mean':[0,0,0],'std':[1,1,1]}
  values,geometry=prepare_profile(image,spec,profile)
  self.assertIsNone(geometry);np.testing.assert_array_equal(values[0,100,100],(np.rint(np.array([20,80,150])/255/.01)-128).astype('i1'))

if __name__=='__main__':unittest.main()
