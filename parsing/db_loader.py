from parsing.jpaf_parser import SHDB_Parser
from parsing.uvafdb_parser import UVAFDB_Parser
from parsing.rbaf_parser import RBAFDB_Parser
from parsing.afdb_parser import AFDB_Parser
from parsing.cpsc_parser import CPSCDB_Parser
from parsing.ltaf_parser import LTAFDB_Parser

PARSER_MAP = {'UVAFDB_Parser': UVAFDB_Parser,
              'JPAFDB_Parser': SHDB_Parser,
              'RBAFDB_Parser': RBAFDB_Parser,
              # 'AFDB_Parser': AFDB_Parser,
              'CPSCDB_Parser': CPSCDB_Parser}
              # 'LTAFDB_Parser': LTAFDB_Parser}