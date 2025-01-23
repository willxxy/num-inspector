from dataclasses import dataclass
import torch
import numpy as np
from typing import Union, Dict, Type
import matplotlib.pyplot as plt

@dataclass
class IntSpecs:
    total_bits: int
    signed: bool
    max_value: int
    min_value: int
    max_unsigned_value: int = None
    bit_pattern_template: str = None

class IntAnalyzer:
    
    SUPPORTED_DTYPES = {
        torch.int8: ("8-bit integer", "PyTorch", True),
        torch.int16: ("16-bit integer", "PyTorch", True),
        torch.int32: ("32-bit integer", "PyTorch", True),
        torch.int64: ("64-bit integer", "PyTorch", True),
        torch.uint8: ("8-bit unsigned integer", "PyTorch", False),
        np.int8: ("8-bit integer", "NumPy", True),
        np.int16: ("16-bit integer", "NumPy", True),
        np.int32: ("32-bit integer", "NumPy", True),
        np.int64: ("64-bit integer", "NumPy", True),
        np.uint8: ("8-bit unsigned integer", "NumPy", False),
        np.uint16: ("16-bit unsigned integer", "NumPy", False),
        np.uint32: ("32-bit unsigned integer", "NumPy", False),
        np.uint64: ("64-bit unsigned integer", "NumPy", False),
        int: ("Python integer (64-bit)", "Python", True)
    }

    def __init__(self):
        self.specs_cache = {}
        self._initialize_specs()

    def _initialize_specs(self):
        for dtype, (_, _, signed) in self.SUPPORTED_DTYPES.items():
            self.specs_cache[dtype] = self._calculate_int_specs(dtype, signed)

    def _calculate_int_specs(self, dtype, signed: bool) -> IntSpecs:
        if dtype == int:
            bits = 64
        elif isinstance(dtype, torch.dtype):
            bits = torch.empty((), dtype=dtype).element_size() * 8
        else:
            bits = np.dtype(dtype).itemsize * 8
        
        if signed:
            max_value = 2**(bits-1) - 1
            min_value = -2**(bits-1)
            max_unsigned = 2**bits - 1
            bit_pattern = "s" + "v" * (bits-1)
        else:
            max_value = 2**bits - 1
            min_value = 0
            max_unsigned = max_value
            bit_pattern = "v" * bits
            
        return IntSpecs(bits, signed, max_value, min_value, max_unsigned, bit_pattern)

    def _get_raw_bytes(self, value: int, specs: IntSpecs) -> bytes:
        if specs.total_bits == 8:
            return value.to_bytes(1, byteorder='big', signed=specs.signed)
        elif specs.total_bits == 16:
            return value.to_bytes(2, byteorder='big', signed=specs.signed)
        elif specs.total_bits == 32:
            return value.to_bytes(4, byteorder='big', signed=specs.signed)
        elif specs.total_bits == 64:
            return value.to_bytes(8, byteorder='big', signed=specs.signed)
        else:
            raise ValueError(f"Unsupported bit width: {specs.total_bits}")

    def analyze_number(self, value: int, dtype: Union[Type[int], torch.dtype, np.dtype]) -> Dict:
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        specs = self.specs_cache[dtype]
        
        try:
            if isinstance(dtype, torch.dtype):
                tensor = torch.tensor(value, dtype=dtype)
                actual_value = tensor.item()
            elif isinstance(dtype, np.dtype):
                array = np.array(value, dtype=dtype)
                actual_value = array.item()
            else:
                actual_value = value
                
            raw_bytes = self._get_raw_bytes(actual_value, specs)
            bin_str = bin(int.from_bytes(raw_bytes, byteorder='big'))[2:].zfill(specs.total_bits)
            
            if specs.signed:
                sign_bit = bin_str[0]
                value_bits = bin_str[1:]
                sign = -1 if sign_bit == '1' else 1
            else:
                sign_bit = None
                value_bits = bin_str
                sign = 1
            
            if specs.signed and sign_bit == '1':
                decimal_value = actual_value
            else:
                decimal_value = int(bin_str, 2)
            
            return {
                'binary': {
                    'full': bin_str,
                    'sign': sign_bit,
                    'value': value_bits,
                    'twos_complement': specs.signed and sign == -1
                },
                'decimal': {
                    'sign': sign,
                    'value': decimal_value,
                    'original_value': value,
                    'overflow': value != actual_value
                },
                'hex': hex(int.from_bytes(raw_bytes, byteorder='big')),
                'specs': specs
            }
            
        except (OverflowError, ValueError) as e:
            return {
                'error': str(e),
                'specs': specs,
                'overflow': True
            }

    def visualize_components(self, value: int, dtype: Union[Type[int], torch.dtype, np.dtype]):
        analysis = self.analyze_number(value, dtype)
        if 'error' in analysis:
            raise ValueError(f"Cannot visualize value {value} for {dtype}: {analysis['error']}")
            
        specs = analysis['specs']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), height_ratios=[3, 1])
        
        if specs.signed:
            components = [
                ('Sign', analysis['binary']['sign'], 'red'),
                ('Value', analysis['binary']['value'], 'blue')
            ]
        else:
            components = [
                ('Value', analysis['binary']['full'], 'blue')
            ]
        
        x_pos = 0
        for name, bits, color in components:
            width = len(bits)
            ax1.add_patch(plt.Rectangle((x_pos, 0), width, 1, facecolor=color, alpha=0.3))
            ax1.text(x_pos + width/2, 0.5, name, ha='center', va='center')
            
            for i, bit in enumerate(bits):
                ax1.text(x_pos + i + width/2, 0.2, bit, ha='center', va='center', fontfamily='monospace')
                ax1.text(x_pos + i + width/2, 0.8, str(specs.total_bits - 1 - (x_pos + i)), 
                        ha='center', va='center', fontsize=8)
            x_pos += width
            
        ax1.set_xlim(0, specs.total_bits)
        ax1.set_ylim(0, 1)
        precision_name = self.SUPPORTED_DTYPES[dtype][0]
        ax1.set_title(f'{precision_name} representation of {value}')
        ax1.axis('off')
        
        info_text = f"Decimal: {analysis['decimal']['value']}"
        if analysis['decimal']['overflow']:
            info_text += f" (Overflow from {analysis['decimal']['original_value']})"
        info_text += f"\nHexadecimal: {analysis['hex']}"
        if specs.signed:
            info_text += f"\nTwo's Complement: {'Yes' if analysis['binary']['twos_complement'] else 'No'}"
        
        ax2.text(0.5, 0.5, info_text, ha='center', va='center', fontsize=10)
        ax2.axis('off')
        
        plt.tight_layout()
        return fig

    def compare_representations(self, value: int):
        results = {}
        for dtype in self.SUPPORTED_DTYPES:
            try:
                if dtype == int:
                    actual_value = value
                elif isinstance(dtype, torch.dtype):
                    tensor = torch.tensor(value, dtype=dtype)
                    actual_value = tensor.item()
                else:
                    array = np.array(value, dtype=dtype)
                    actual_value = array.item()
                
                results[dtype] = {
                    'storage': self.analyze_number(actual_value, dtype),
                    'actual_value': actual_value,
                    'overflow': actual_value != value
                }
            except (OverflowError, ValueError) as e:
                results[dtype] = {'error': str(e)}
                
        return results

    def analyze_special_values(self, dtype: Union[Type[int], torch.dtype, np.dtype]):
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        specs = self.specs_cache[dtype]
        special_values = {
            'zero': 0,
            'max_value': specs.max_value,
            'min_value': specs.min_value,
            'minus_one': -1 if specs.signed else None,
            'max_unsigned_value': specs.max_unsigned_value,
            'boundary_value': specs.max_value // 2,
            'boundary_plus_one': (specs.max_value // 2) + 1
        }
        
        results = {}
        for name, value in special_values.items():
            if value is not None:
                try:
                    results[name] = self.analyze_number(value, dtype)
                except (OverflowError, ValueError, RuntimeError) as e:
                    results[name] = {
                        'error': f"Overflow: {value} is outside the valid range for {self.SUPPORTED_DTYPES[dtype][0]}",
                        'specs': specs,
                        'overflow': True
                    }
                
        return results

def demo():
    analyzer = IntAnalyzer()
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--value', type=int, default=11)
    args = parser.parse_args()
    
    value = args.value
    print(f"\nAnalyzing {value} across different integer representations:")
    results = analyzer.compare_representations(value)
    
    def get_type_name(dtype):
        if dtype == int:
            return 'int'
        elif isinstance(dtype, (torch.dtype, np.dtype)):
            return str(dtype).split('.')[-1]
        return str(dtype)
    
    for dtype, result in results.items():
        precision_name, framework, _ = analyzer.SUPPORTED_DTYPES[dtype]
        print(f"\n{precision_name} ({framework}):")
        if 'error' in result:
            print(f"Error: {result['error']}")
        else:
            print(f"Actual value: {result['actual_value']}")
            print(f"Overflow occurred: {result['overflow']}")
            print(f"Binary: {result['storage']['binary']['full']}")
            print(f"Hex: {result['storage']['hex']}")
    
    print("\nAnalyzing special values across all supported types:")
    for dtype in analyzer.SUPPORTED_DTYPES:
        precision_name, framework, _ = analyzer.SUPPORTED_DTYPES[dtype]
        print(f"\n=== {precision_name} ({framework}) Special Values ===")
        special_results = analyzer.analyze_special_values(dtype)
        
        for name, result in special_results.items():
            print(f"\n{name}:")
            if 'error' in result:
                print(f"Error: {result['error']}")
            else:
                print(f"Binary: {result['binary']['full']}")
                print(f"Hex: {result['hex']}")
    
    print("\nGenerating visualizations for 42 across all supported types...")
    for dtype in analyzer.SUPPORTED_DTYPES:
        precision_name, framework, _ = analyzer.SUPPORTED_DTYPES[dtype]
        try:
            fig = analyzer.visualize_components(value, dtype)
            plt.savefig(f'./pngs/int_components_{framework.lower()}_{get_type_name(dtype)}_42.png')
            plt.close()
            print(f"Generated visualization for {precision_name} ({framework})")
        except (ValueError, OverflowError, RuntimeError) as e:
            print(f"Could not visualize {precision_name}: Value {value} is outside the valid range")

if __name__ == "__main__":
    demo() 