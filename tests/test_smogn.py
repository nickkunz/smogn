import pytest
import pandas as pd
import smogn

@pytest.fixture
def housing_data():
    return pd.read_csv(
    ## http://jse.amstat.org/v19n3/decock.pdf
    'https://raw.githubusercontent.com/nickkunz/smogn/master/data/housing.csv'
)
    
# content of test_class.py
class TestExampleBeginner:
    """ Integration test drawn from examples/smogn_example_1_beg.ipynb
    """
    def test_housing_data(self, housing_data):
        print("Use the same housing dataset")
        assert housing_data.shape == (1460, 81)
        
        
    def test_example1(self, housing_data):
        print("smoter example should pass")
        housing_data
        ## conduct smogn
        housing_smogn = smogn.smoter(   
            data = housing_data,  ## pandas dataframe
            y = 'SalePrice'  ## string ('header name')
        )
        assert housing_smogn.shape == (1244, 62)