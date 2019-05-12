import unittest
import GAMITArchive


class ConnectionTest(unittest.TestCase):

    def setUp(self) -> None:
        self.config_file = 'debug.cfg'
        self.stationcode = 'test'

    def test_Cnn(self) -> None:
        cnn = GAMITArchive.Connection(self.config_file)
        self.assertEqual(cnn.conn.closed, 0)


if __name__ == '__main__':
    unittest.main()
