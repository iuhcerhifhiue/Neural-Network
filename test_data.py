import unittest
import os
import torch
from unittest.mock import patch, mock_open, MagicMock
from data import CharTokenizer, load_text, build_dataset, get_batch

# Mock urllib.request for load_text tests
def mock_urlretrieve(url, filename):
    # Simulate downloading and writing content to the file
    with open(filename, "w") as f:
        f.write("mock content")

class TestCharTokenizer(unittest.TestCase):
    def setUp(self):
        self.text = "hello world"
        self.tokenizer = CharTokenizer(self.text)

    def test_vocab_size(self):
        expected_chars = sorted(list(set(self.text)))
        self.assertEqual(self.tokenizer.vocab_size, len(expected_chars))

    def test_stoi(self):
        self.assertIn('h', self.tokenizer.stoi)
        self.assertEqual(self.tokenizer.stoi[' '], 0) # Assuming space is the first char after sorting

    def test_itos(self):
        self.assertIn(0, self.tokenizer.itos)
        self.assertEqual(self.tokenizer.itos[0], ' ')

    def test_encode(self):
        encoded = self.tokenizer.encode("hello")
        self.assertIsInstance(encoded, list)
        self.assertEqual(len(encoded), 5)
        self.assertEqual(self.tokenizer.decode(encoded), "hello")

    def test_decode(self):
        decoded = self.tokenizer.decode([self.tokenizer.stoi['h'], self.tokenizer.stoi['e'], self.tokenizer.stoi['l'], self.tokenizer.stoi['l'], self.tokenizer.stoi['o']])
        self.assertEqual(decoded, "hello")

    def test_encode_decode_roundtrip(self):
        original_text = "test string with various characters 123!"
        tokenizer = CharTokenizer(original_text)
        encoded = tokenizer.encode(original_text)
        decoded = tokenizer.decode(encoded)
        self.assertEqual(decoded, original_text)

class TestLoadText(unittest.TestCase):
    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    @patch('urllib.request.urlretrieve', side_effect=mock_urlretrieve)
    @patch('builtins.open', new_callable=mock_open, read_data="downloaded content")
    def test_downloads_if_not_exists(self, mock_file_open, mock_urlretrieve, mock_exists, mock_makedirs):
        content = load_text(data_dir="./test_data_dir")
        mock_makedirs.assert_called_once_with("./test_data_dir", exist_ok=True)
        mock_exists.assert_called_once()
        mock_urlretrieve.assert_called_once()
        mock_file_open.assert_called_once_with(os.path.join("./test_data_dir", "input.txt"), "r", encoding="utf-8")
        self.assertEqual(content, "downloaded content")

    @patch('os.makedirs')
    @patch('os.path.exists', return_value=True)
    @patch('urllib.request.urlretrieve')
    @patch('builtins.open', new_callable=mock_open, read_data="existing content")
    def test_reads_if_exists(self, mock_file_open, mock_urlretrieve, mock_exists, mock_makedirs):
        content = load_text(data_dir="./test_data_dir")
        mock_makedirs.assert_called_once_with("./test_data_dir", exist_ok=True)
        mock_exists.assert_called_once()
        mock_urlretrieve.assert_not_called()
        mock_file_open.assert_called_once_with(os.path.join("./test_data_dir", "input.txt"), "r", encoding="utf-8")
        self.assertEqual(content, "existing content")

    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    @patch('urllib.request.urlretrieve', side_effect=urllib.error.URLError('test error'))
    @patch('builtins.open', new_callable=mock_open)
    def test_download_error(self, mock_file_open, mock_urlretrieve, mock_exists, mock_makedirs):
        with self.assertRaises(IOError):
            load_text(data_dir="./test_data_dir")

    @patch('os.makedirs')
    @patch('os.path.exists', return_value=False)
    @patch('urllib.request.urlretrieve', side_effect=urllib.error.HTTPError('http://example.com', 404, 'Not Found', {}, None))
    @patch('builtins.open', new_callable=mock_open)
    def test_download_http_error(self, mock_file_open, mock_urlretrieve, mock_exists, mock_makedirs):
        with self.assertRaises(IOError):
            load_text(data_dir="./test_data_dir")

class TestBuildDataset(unittest.TestCase):
    @patch('data.load_text', return_value="abcde")
    def test_build_dataset(self, mock_load_text):
        tokenizer, train_data, val_data = build_dataset(val_frac=0.2)

        mock_load_text.assert_called_once()
        self.assertIsInstance(tokenizer, CharTokenizer)
        self.assertEqual(tokenizer.vocab_size, 5) # a, b, c, d, e

        # 'abcde', val_frac=0.2 -> 5 * 0.8 = 4 for train, 1 for val
        self.assertEqual(train_data.shape[0], 4)
        self.assertEqual(val_data.shape[0], 1)

        self.assertEqual(tokenizer.decode(train_data.tolist()), "abcd")
        self.assertEqual(tokenizer.decode(val_data.tolist()), "e")

    @patch('data.load_text', return_value="abc")
    def test_build_dataset_no_val_frac(self, mock_load_text):
        tokenizer, train_data, val_data = build_dataset(val_frac=0.0)
        self.assertEqual(train_data.shape[0], 3)
        self.assertEqual(val_data.shape[0], 0)

class TestGetBatch(unittest.TestCase):
    def setUp(self):
        self.data = torch.tensor([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=torch.long)
        self.block_size = 4
        self.batch_size = 2
        self.device = 'cpu'

    def test_get_batch_output_shapes(self):
        x, y = get_batch(self.data, self.block_size, self.batch_size, self.device)
        self.assertEqual(x.shape, (self.batch_size, self.block_size))
        self.assertEqual(y.shape, (self.batch_size, self.block_size))
        self.assertEqual(x.device.type, self.device)
        self.assertEqual(y.device.type, self.device)

    def test_get_batch_shifted_targets(self):
        # For a small, deterministic data, we can check the shift
        # Since torch.randint is used, we can't easily predict the exact batch, but we can verify the shift property
        x, y = get_batch(self.data, self.block_size, 1, self.device) # single batch for easier inspection
        self.assertEqual(x[0, 1:], y[0, :-1]) # input shifted by one to get target

    def test_get_batch_value_error(self):
        with self.assertRaisesRegex(ValueError, "Data length \(.*\) must be greater than block_size \(.*\) to sample a batch."):
            get_batch(self.data[:self.block_size - 1], self.block_size, self.batch_size, self.device) # data too short

