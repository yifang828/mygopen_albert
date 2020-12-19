import unittest
from preprocess.lt_udn_dataset import LtUdnDataset
from preprocess.ll_ncl_ps_dataset import LlNclPs
# from preprocess.ll_ncl_ps_x_dataset import LlNclPsX
# from preprocess.ll_ncl_others_ps_dataset import LlNclOthersPs
# from preprocess.ll_ncl_others_ps_x_dataset import LlNclOthersPsX
from transformers import *

class LtUdnDatasetTestCase(unittest.TestCase):
    def setUp(self):
        pass
    def tearDown(self):
        pass
    def test_getTrainData(self):
        data = LtUdnDataset()
        train_df = data.get_train_dataset()
        self.assertEqual((2257, 3), train_df.shape)
    def test_getTestData(self):
        data = LtUdnDataset()
        test_df = data.get_test_dataset()
        self.assertEqual((564, 3), test_df.shape)
    def test_getLlNclPs(self):
        data = LlNclPs('train', BertTokenizer)
        data.get_required_label_dataset()
        print(data.df['label'].unique())
        # expected = ['PS' 'NCL' 'LL']
        self.assertEqual(3, len(data.df['label'].unique()))
        
if __name__ == '__main__':
    unittest.main()