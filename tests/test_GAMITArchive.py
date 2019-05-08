import unittest
import GAMITArchive


class ConnectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config_file = 'debug.cfg'
        self.stationcode = 'test'

    def test_Cnn(self):
        cnn = GAMITArchive.Connection(self.config_file)


if __name__ == '__main__':
    unittest.main()
