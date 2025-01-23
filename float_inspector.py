import os
import torch
import numpy as np
import struct
import binascii
from typing import Union, Dict, Type
import matplotlib.pyplot as plt
from dataclasses import dataclass

@dataclass
class FloatSpecs:
    total_bits: int
    exponent_bits: int
    mantissa_bits: int
    bias: int
    max_value: float
    min_value: float
    epsilon: float

class FloatAnalyzer:
    
    SUPPORTED_DTYPES = {
        float: ("Python float (64-bit)", "Python"),
        torch.float16: ("Half precision (16-bit)", "PyTorch"),
        torch.float32: ("Single precision (32-bit)", "PyTorch"),
        torch.float64: ("Double precision (64-bit)", "PyTorch"),
        np.float16: ("Half precision (16-bit)", "NumPy"),
        np.float32: ("Single precision (32-bit)", "NumPy"),
        np.float64: ("Double precision (64-bit)", "NumPy")
    }

    def __init__(self):
        self.specs_cache = {}
        self._initialize_specs()

    def _initialize_specs(self):
        self.specs_cache[float] = self._calculate_float_specs(float)
        
        for dtype in [torch.float16, torch.float32, torch.float64]:
            self.specs_cache[dtype] = self._calculate_float_specs(dtype)
        for dtype in [np.float16, np.float32, np.float64]:
            self.specs_cache[dtype] = self._calculate_float_specs(dtype)

    def _calculate_float_specs(self, dtype) -> FloatSpecs:
        if dtype == float or dtype in [torch.float64, np.float64]:
            bits, exp_bits, mantissa_bits = 64, 11, 52
            max_value = 1.7976931348623157e+308
            min_value = 2.2250738585072014e-308
            epsilon = 2.220446049250313e-16
        elif dtype in [torch.float16, np.float16]:
            bits, exp_bits, mantissa_bits = 16, 5, 10
            max_value = 65504.0
            min_value = 6.103515625e-05
            epsilon = 0.0009765625
        elif dtype in [torch.float32, np.float32]:
            bits, exp_bits, mantissa_bits = 32, 8, 23
            max_value = 3.4028235e+38
            min_value = 1.1754944e-38
            epsilon = 1.1920929e-07
        else:
            raise ValueError(f"Unsupported dtype: {dtype}")

        bias = 2**(exp_bits-1) - 1
        
        return FloatSpecs(bits, exp_bits, mantissa_bits, bias, max_value, min_value, epsilon)

    def _get_raw_bytes(self, value: float, specs: FloatSpecs) -> bytes:
        if specs.total_bits == 16:
            f32_bytes = struct.pack('!f', float(value))
            return f32_bytes[:2]
        elif specs.total_bits == 32:
            return struct.pack('!f', float(value))
        elif specs.total_bits == 64:
            return struct.pack('!d', float(value))
        else:
            raise ValueError(f"Unsupported bit width: {specs.total_bits}")

    def analyze_number(self, value: float, dtype: Union[Type[float], torch.dtype, np.dtype]) -> Dict:
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")

        specs = self.specs_cache[dtype]
        raw_bytes = self._get_raw_bytes(value, specs)
        
        bin_str = bin(int.from_bytes(raw_bytes, byteorder='big'))[2:].zfill(specs.total_bits)
        
        sign_bit = bin_str[0]
        exponent = bin_str[1:specs.exponent_bits+1]
        mantissa = bin_str[specs.exponent_bits+1:]
        
        sign = -1 if sign_bit == '1' else 1
        exp_val = int(exponent, 2) - specs.bias
        mantissa_val = 1 + sum(int(b) * 2**-(i+1) for i, b in enumerate(mantissa))
        
        return {
            'binary': {
                'full': bin_str,
                'sign': sign_bit,
                'exponent': exponent,
                'mantissa': mantissa
            },
            'decimal': {
                'sign': sign,
                'exponent': exp_val,
                'mantissa': mantissa_val,
                'value': sign * mantissa_val * (2.0 ** exp_val)
            },
            'hex': binascii.hexlify(raw_bytes).decode(),
            'specs': specs
        }

    def compare_representations(self, value: float):
        results = {}
        for dtype in self.SUPPORTED_DTYPES:
            try:
                if dtype == float:
                    actual_value = float(value)
                elif isinstance(dtype, torch.dtype):
                    tensor = torch.tensor(value, dtype=dtype)
                    actual_value = tensor.item()
                else:
                    array = np.array(value, dtype=dtype)
                    actual_value = array.item()
                
                results[dtype] = {
                    'storage': self.analyze_number(actual_value, dtype),
                    'actual_value': actual_value,
                    'relative_error': abs(actual_value - value) / abs(value) if value != 0 else abs(actual_value),
                    'absolute_error': abs(actual_value - value)
                }
            except (OverflowError, ValueError) as e:
                results[dtype] = {'error': str(e)}
                
        return results

    def visualize_components(self, value: float, dtype: Union[Type[float], torch.dtype, np.dtype]):
        analysis = self.analyze_number(value, dtype)
        specs = analysis['specs']
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 5), height_ratios=[3, 1])
        
        components = [
            ('Sign', analysis['binary']['sign'], 'red'),
            ('Exponent', analysis['binary']['exponent'], 'green'),
            ('Mantissa', analysis['binary']['mantissa'], 'blue')
        ]
        
        x_pos = 0
        for name, bits, color in components:
            width = len(bits)
            ax1.add_patch(plt.Rectangle((x_pos, 0), width, 1, facecolor=color, alpha=0.3))
            ax1.text(x_pos + width/2, 0.5, name, ha='center', va='center')
            ax1.text(x_pos + width/2, 0.2, bits, ha='center', va='center', fontfamily='monospace')
            x_pos += width
        
        ax1.set_xlim(0, specs.total_bits)
        ax1.set_ylim(0, 1)
        precision_name = self.SUPPORTED_DTYPES[dtype][0]
        ax1.set_title(f'{precision_name} representation of {value}')
        ax1.axis('off')
        
        formula = f"{analysis['decimal']['sign']} × {analysis['decimal']['mantissa']:.10f} × 2^{analysis['decimal']['exponent']}"
        ax2.text(0.5, 0.5, f"Value = {formula} = {analysis['decimal']['value']}", 
                ha='center', va='center', fontsize=10)
        ax2.axis('off')
        
        plt.tight_layout()
        return fig

    def analyze_special_values(self, dtype: Union[Type[float], torch.dtype, np.dtype]):
        if dtype not in self.SUPPORTED_DTYPES:
            raise ValueError(f"Unsupported dtype: {dtype}")
            
        special_values = {
            'zero': 0.0,
            'infinity': float('inf'),
            'negative_infinity': float('-inf'),
            'nan': float('nan'),
            'min_positive': self.specs_cache[dtype].min_value,
            'max_finite': self.specs_cache[dtype].max_value,
            'epsilon': self.specs_cache[dtype].epsilon,
            'negative_zero': -0.0
        }
        
        results = {}
        for name, value in special_values.items():
            try:
                results[name] = self.analyze_number(value, dtype)
            except (OverflowError, ValueError) as e:
                results[name] = {'error': str(e)}
                
        return results

def demo():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--value', type=float, default=3.14159)
    args = parser.parse_args()
    
    analyzer = FloatAnalyzer()
    
    value = args.value
    print(f"\nAnalyzing {value} across different representations:")
    results = analyzer.compare_representations(value)
    
    for dtype, result in results.items():
        if 'error' not in result:
            precision_name, framework = analyzer.SUPPORTED_DTYPES[dtype]
            print(f"\n{precision_name} ({framework}):")
            print(f"Actual value: {result['actual_value']}")
            print(f"Relative error: {result['relative_error']:.2e}")
            print(f"Absolute error: {result['absolute_error']:.2e}")
            print(f"Binary: {result['storage']['binary']['full']}")
            print(f"Hex: {result['storage']['hex']}")
    
    def get_type_name(dtype):
        if dtype == float:
            return 'float'
        elif isinstance(dtype, (torch.dtype, np.dtype)):
            return str(dtype).split('.')[-1]
        return str(dtype)

    print("\nAnalyzing special values across all supported types:")
    for dtype in analyzer.SUPPORTED_DTYPES:
        precision_name, framework = analyzer.SUPPORTED_DTYPES[dtype]
        print(f"\n=== {precision_name} ({framework}) Special Values ===")
        special_results = analyzer.analyze_special_values(dtype)
        
        for name, result in special_results.items():
            if 'error' not in result:
                print(f"\n{name}:")
                print(f"Binary: {result['binary']['full']}")
                print(f"Hex: {result['hex']}")
                
    
    os.makedirs('./pngs', exist_ok=True)
    print("\nGenerating visualizations for π across all supported types...")
    for dtype in analyzer.SUPPORTED_DTYPES:
        precision_name, framework = analyzer.SUPPORTED_DTYPES[dtype]
        try:
            fig = analyzer.visualize_components(value, dtype)
            plt.savefig(f'./pngs/float_components_{framework.lower()}_{get_type_name(dtype)}_pi.png')
            plt.close()
            print(f"Generated visualization for {precision_name} ({framework})")
        except (ValueError, OverflowError) as e:
            print(f"Could not visualize {precision_name}: {e}")

if __name__ == "__main__":
    demo()