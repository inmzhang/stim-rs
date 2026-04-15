use super::{Result, StimError};
use ndarray::{Array2, ArrayView2};

pub fn packed_row_bytes(bit_len: usize) -> usize {
    bit_len.div_ceil(8)
}

pub fn unpack_bits(bytes: &[u8], bit_len: usize) -> Vec<bool> {
    let mut result = Vec::with_capacity(bit_len);
    for bit_index in 0..bit_len {
        let byte = bytes[bit_index / 8];
        result.push(((byte >> (bit_index % 8)) & 1) != 0);
    }
    result
}

fn unpack_rows_vec(bytes: &[u8], bit_len: usize) -> Vec<Vec<bool>> {
    let row_bytes = packed_row_bytes(bit_len);
    bytes
        .chunks(row_bytes)
        .map(|row| unpack_bits(row, bit_len))
        .collect()
}

pub(crate) fn unpack_rows_array(bytes: &[u8], bit_len: usize) -> Array2<bool> {
    let rows = unpack_rows_vec(bytes, bit_len);
    let nrows = rows.len();
    let ncols = rows.first().map_or(0, Vec::len);
    Array2::from_shape_vec((nrows, ncols), rows.into_iter().flatten().collect())
        .expect("packed rows should decode into a rectangular array")
}

pub fn pack_bits(bits: &[bool]) -> Vec<u8> {
    let mut packed = vec![0u8; packed_row_bytes(bits.len())];
    for (bit_index, value) in bits.iter().copied().enumerate() {
        if value {
            packed[bit_index / 8] |= 1 << (bit_index % 8);
        }
    }
    packed
}

fn pack_rows_vec(rows: &[Vec<bool>], bit_len: usize) -> Result<Vec<u8>> {
    let mut packed = Vec::with_capacity(rows.len() * packed_row_bytes(bit_len));
    for row in rows {
        if row.len() != bit_len {
            return Err(StimError::new(format!(
                "expected {bit_len} bits per shot, got {}",
                row.len()
            )));
        }
        packed.extend(pack_bits(row));
    }
    Ok(packed)
}

pub(crate) fn pack_rows_array(rows: ArrayView2<'_, bool>, bit_len: usize) -> Result<Vec<u8>> {
    let borrowed = rows
        .rows()
        .into_iter()
        .map(|row| row.to_vec())
        .collect::<Vec<_>>();
    pack_rows_vec(&borrowed, bit_len)
}
