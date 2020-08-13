import logging
import unittest

from pint import UnitRegistry

from eam_core.dsl.check_visitor import evaluate

logger = logging.getLogger(__name__)

ureg = UnitRegistry(auto_reduce_dimensions=False)
# ureg.define('bit = [information]')
ureg.default_format = '~H'
Q_ = ureg.Quantity


class NumberTestCase(unittest.TestCase):

    def test_new_assignment_not_implicit(self):
        block = """
            a = 3 >= 1;
            a = false;
            b = a + 1;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'a' not in visitor.implicit_variables
        assert 'b' in visitor.new_variables
        assert len(visitor.new_variables) == 2
        assert len(visitor.implicit_variables) == 0

    def test_implicit_var_(self):
        block = """
            b = a + 1;
            a = 3;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.implicit_variables
        assert 'a' not in visitor.new_variables
        assert 'b' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 1

    def test_basic_int_assign(self):
        block = """
            a = 2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_repeat_int_assign(self):
        block = """
            a = 2;
            a = 3;
            a = a;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_float_assign(self):
        block = """
            a = 2.2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_str_assign(self):
        block = """
            a = "hello";
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_self_assign(self):
        block = """
            a = 1 + a;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.implicit_variables
        assert len(visitor.new_variables) == 0
        assert len(visitor.implicit_variables) == 1

    def test_2nd_basic_self_assign(self):
        block = """
            a = a+1;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.implicit_variables
        assert len(visitor.new_variables) == 0
        assert len(visitor.implicit_variables) == 1

    def test_basic_nil_assign(self):
        block = """
            a = nil;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_add(self):
        block = """
            a = 2+2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_sub(self):
        block = """
            a = 6-2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_several_add(self):
        block = """
            a = 1 + 1 + 1 + 1;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_unary_minus(self):
        block = """
            a = -2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_mult(self):
        block = """
            a = 2*2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_div(self):
        block = """
            a = 8/2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_mod(self):
        block = """
            a = 14 % 5;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_pow(self):
        block = """
            a = 2^2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_paranthesis(self):
        block = """
            a = (2+2)*2;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_bool_true(self):
        block = """
            a = true;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_bool_false(self):
        block = """
            a = false;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_bool_not(self):
        block = """
            a = !false;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_eq(self):
        block = """
            a = 3 == 1;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables

    def test_basic_gt(self):
        block = """
            a = 3 > 1;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_get(self):
        block = """
            a = 3 >= 1;
            b = 3 >= 3;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.new_variables
        assert len(visitor.new_variables) == 2
        assert len(visitor.implicit_variables) == 0

    def test_basic_let(self):
        block = """
            a = 1 <= 1;
            b = 1 <= 3;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 2
        assert len(visitor.implicit_variables) == 0

    def test_basic_lt(self):
        block = """
            a = 1 < 3;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 0

    def test_basic_and(self):
        block = """
            a = true && true;
            b = (2<3) &&  ( 4 >= 4);
            c = false && true;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.new_variables
        assert 'c' in visitor.new_variables
        assert len(visitor.new_variables) == 3
        assert len(visitor.implicit_variables) == 0

    def test_basic_or(self):
        block = """
            a = true || false;
            b = (2 <3 ) ||  ( 4 >= 1);
            c = false || false;
        """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.new_variables
        assert 'c' in visitor.new_variables
        assert len(visitor.new_variables) == 3
        assert len(visitor.implicit_variables) == 0

    def test_basic_if(self):
        block = """
                   a = true ;
                   if (a){
                        b = 2;
                   }
               """
        visitor = evaluate(block)

        assert 'a' in visitor.new_variables
        assert 'b' in visitor.new_variables

        assert len(visitor.new_variables) == 2
        assert len(visitor.implicit_variables) == 0

    def test_basic_if_conditions_implicit(self):
        """
        test that implicit atoms in conditions are found
        :return:
        """
        block = """
                   if (a){
                        b = 2;
                   }
               """
        visitor = evaluate(block)

        assert 'a' in visitor.implicit_variables
        assert 'b' in visitor.new_variables

        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 1

    def test_basic_if_implicit_stat(self):
        block = """
                   a=2;
                   if (a){
                        a = b;
                   }
               """
        visitor = evaluate(block)

        assert 'a' in visitor.new_variables
        assert 'b' in visitor.implicit_variables

        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 1

    def test_basic_if_else_if_else(self):
        block = """
                   a = 0 ;
                   if (a != 0){
                        b = 2;
                   } else if (b != 0){
                        c = d;
                   } else {
                        a = c;
                   }
               """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.implicit_variables
        assert 'c' in visitor.implicit_variables
        assert 'd' in visitor.implicit_variables
        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 3

    def test_model_formula_if_else_if_else(self):
        block = """
      if (measured_carbon_CDN != 0){
        carbon = measured_carbon_CDN;
      } else if (measured_energy_CDN  != 0) {
        energy = measured_energy_CDN;
        x = test;
        carbon = energy * carbon_intensity_uk;
      } else {
        energy = energy_intensity_cdn * total_fixed_line_data_volume_per_ref_duration;
        carbon = energy * carbon_intensity_uk;
      }
               """
        visitor = evaluate(block)

        assert 'measured_energy_CDN' in visitor.implicit_variables

    def test_basic_any_implicit_in_stat_block_found(self):
        """
        test that any implicit variable in a stat block is found, even if an assigment in another block occurs
        :return:
        """
        block = """
                   a = 0 ;
                   if (a){
                        b = 2;
                   } else {
                        a = b;
                   }
               """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.implicit_variables

        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 1

    def test_basic_if_else(self):
        block = """
                   a = true ;
                   if (a){
                        b = 2;
                   } else {
                        c = false;
                   }
               """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.new_variables
        assert 'c' in visitor.new_variables
        assert len(visitor.new_variables) == 3
        assert len(visitor.implicit_variables) == 0

    def test_basic_while(self):
        block = """
                   a = 10;
                   while (c > 0){
                        b= b+1;
                        a = a- 1;
                   }
               """
        visitor = evaluate(block)
        assert 'a' in visitor.new_variables
        assert 'b' in visitor.implicit_variables
        assert 'c' in visitor.implicit_variables

        assert len(visitor.new_variables) == 1
        assert len(visitor.implicit_variables) == 2


if __name__ == '__main__':
    unittest.main()
