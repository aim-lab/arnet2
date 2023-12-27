
# Parsing

Collection of pasrers supporting a set of databases. Here, `BaseParser` is defined in the module base_parser.py. All other .py files inherit from `BaseParser` extending the list of variables and the parser functions.

### Supported databases

| Database  | type         | Parser          | .py file          | Description|
|:----------|:-------------|:--------------  |:------------------|:-----------|
| uvfdb/uvaf|Holter        |`UVAFDB_Parser`  | uvafdb_parser.py  | private
| rbdb      |Holter        |`RBDB_Parser`    | rbaf_parser.py    | private
| shdb      |Holter        |`SHDB_Parser`    | shdb_parser.py    | private
| afdb      |Holter        |`AFDB_Parser`    | afdb_parser.py    | [link](https://paperswithcode.com/dataset/mit-bih-afdb)|
| cpsc      |Holter        |`CPSCDB_Parser`  | cpsc_parser.py    | [link](http://www.icbeb.org/CPSC2021)
| ltaf      |Holter        |`LTAFDB_Parser`  | ltaf_parser.py    | [link](https://physionet.org/content/ltafdb/1.0.0/)|
| mesa      |Holter        |`MESA_Parser`    | mesa_parser.py    | [link](https://sleepdata.org/datasets/mesa/pages/hrv-analysis.md)
| cfs       |Holter        |`CFS_Parser`     | cfs_parser.py     | [link](https://sleepdata.org/datasets/cfs)
| shhs      |Holter        |`SHHS_Parser`    | shhs_parser.py    | [link](https://sleepdata.org/datasets/shhs/pages/04-dataset-introduction.md)
| spafda    |7-days Holter |`SPAFDB_Parser`  | spafda_parser.py  | private
| fah7db    |7-days Holter |`FAH7DB_Parser`  | fah7db_parser.py  | private
| shdb_2wk  |2-weeks Holter|`SHDB_2wk_Parser`| shdb_2wk_parser.py| private

