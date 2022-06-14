import simplejson as json
import os
import unittest


class MyTestCase(unittest.TestCase):
    def test_expand_json(self):
        os.environ["DEBUSSY"] = "1"
        json_str = '{"key":"$DEBUSSY"}'
        expanded_json = os.path.expandvars(json_str)
        json_o = json.loads(expanded_json)
        assert json_o['key'] == '1'



if __name__ == '__main__':
    unittest.main()
