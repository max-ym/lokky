use core::{char::decode_utf16, convert::Infallible, mem::MaybeUninit, ptr, str::Utf8Error};
use core::slice;

use crate::{
    scope::AllocMarker,
    vec::{TryReserveError, Vec},
    ArrayAllocError,
};

pub struct String {
    vec: Vec<u8>,
}

impl String {
    #[inline]
    pub fn new() -> Self {
        String { vec: Vec::new() }
    }

    #[inline]
    pub fn with_capacity(capacity: usize) -> Self {
        String {
            vec: Vec::with_capacity(capacity),
        }
    }

    #[inline]
    pub fn try_with_capacity(capacity: usize) -> Result<Self, ArrayAllocError> {
        Ok(String {
            vec: Vec::try_with_capacity(capacity)?,
        })
    }

    #[inline]
    pub fn with_marker(marker: AllocMarker) -> Self {
        String {
            vec: Vec::with_marker(marker),
        }
    }

    #[inline]
    pub fn with_capacity_and_marker(capacity: usize, marker: AllocMarker) -> Self {
        String {
            vec: Vec::with_capacity_and_marker(capacity, marker),
        }
    }

    #[inline]
    pub fn try_with_capacity_and_marker(
        capacity: usize,
        marker: AllocMarker,
    ) -> Result<Self, ArrayAllocError> {
        Ok(String {
            vec: Vec::try_with_capacity_and_marker(capacity, marker)?,
        })
    }

    #[inline]
    pub fn len(&self) -> usize {
        self.vec.len()
    }

    #[inline]
    pub fn capacity(&self) -> usize {
        self.vec.capacity()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.vec.is_empty()
    }

    #[inline]
    pub fn try_reserve(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve(additional)
    }

    #[inline]
    pub fn reserve(&mut self, additional: usize) {
        self.vec.reserve(additional)
    }

    #[inline]
    pub fn try_reserve_exact(&mut self, additional: usize) -> Result<(), TryReserveError> {
        self.vec.try_reserve_exact(additional)
    }

    #[inline]
    pub fn reserve_exact(&mut self, additional: usize) {
        self.vec.reserve_exact(additional)
    }

    #[inline]
    pub fn shrink_to(&mut self, min_capacity: usize) {
        self.vec.shrink_to(min_capacity)
    }

    #[inline]
    pub fn shrink_to_fit(&mut self) {
        self.vec.shrink_to_fit()
    }

    #[inline]
    pub fn as_bytes(&self) -> &[u8] {
        &self.vec
    }

    #[inline]
    pub fn as_mut_vec(&mut self) -> &mut Vec<u8> {
        &mut self.vec
    }

    #[inline]
    pub fn as_str(&self) -> &str {
        self
    }

    #[inline]
    pub fn as_mut_str(&mut self) -> &mut str {
        self
    }

    #[inline]
    pub fn push(&mut self, ch: char) {
        match ch.len_utf8() {
            1 => self.vec.push(ch as u8),
            _ => self
                .vec
                .extend_from_slice(ch.encode_utf8(&mut unsafe { uninit_bits() }).as_bytes()),
        }
    }

    #[inline]
    pub fn pop(&mut self) -> Option<char> {
        let ch = self.chars().rev().next()?;
        let new_len = self.len() - ch.len_utf8();
        unsafe { self.vec.set_len(new_len) };
        Some(ch)
    }

    #[inline]
    pub fn push_str(&mut self, string: &str) {
        self.vec.extend_from_slice(string.as_bytes())
    }

    #[inline]
    pub fn clear(&mut self) {
        self.vec.clear();
    }

    #[inline]
    pub fn remove(&mut self, idx: usize) -> char {
        let ch = match self[idx..].chars().next() {
            Some(ch) => ch,
            None => panic!("cannot remove a char from the end of a string"),
        };

        let next = idx + ch.len_utf8();
        let len = self.len();
        unsafe {
            ptr::copy(
                self.vec.as_ptr().add(next),
                self.vec.as_mut_ptr().add(idx),
                len - next,
            );
            self.vec.set_len(len - (next - idx));
        }
        ch
    }

    #[inline]
    pub fn insert(&mut self, idx: usize, ch: char) {
        assert!(self.is_char_boundary(idx));

        let mut bits = unsafe { uninit_bits() };
        let bits = ch.encode_utf8(&mut bits).as_bytes();
        unsafe { self.insert_bytes(idx, bits) };
    }

    #[inline]
    pub fn insert_str(&mut self, idx: usize, string: &str) {
        assert!(self.is_char_boundary(idx));

        unsafe { self.insert_bytes(idx, string.as_bytes()) };
    }

    unsafe fn insert_bytes(&mut self, idx: usize, bytes: &[u8]) {
        let len = self.len();
        let amt = bytes.len();
        self.vec.reserve(amt);

        ptr::copy(
            self.vec.as_ptr().add(idx),
            self.vec.as_mut_ptr().add(idx + amt),
            len - idx,
        );
        ptr::copy_nonoverlapping(bytes.as_ptr(), self.vec.as_mut_ptr().add(idx), amt);
        self.vec.set_len(len + amt);
    }

    #[inline]
    pub fn truncate(&mut self, new_len: usize) {
        if new_len < self.len() {
            assert!(self.is_char_boundary(new_len));
            self.vec.truncate(new_len);
        }
    }

    #[inline]
    #[must_use = "use `.truncate()` if you don't need the other half"]
    pub fn split_off(&mut self, at: usize) -> String {
        assert!(self.is_char_boundary(at));
        let other = self.vec.split_off(at);
        unsafe { String::from_utf8_unchecked(other) }
    }

    #[inline]
    #[must_use]
    pub unsafe fn from_utf8_unchecked(bytes: Vec<u8>) -> String {
        String { vec: bytes }
    }

    #[inline]
    pub fn from_utf8(bytes: Vec<u8>) -> Result<String, Utf8Error> {
        match core::str::from_utf8(&bytes) {
            Ok(..) => Ok(String { vec: bytes }),
            Err(e) => Err(e),
        }
    }

    pub fn from_utf16(v: &[u16]) -> Result<String, Utf16Error> {
        // This isn't done via collect::<Result<_, _>>() for performance reasons.
        // FIXME: the function can be simplified again when #48994 is closed.
        let mut ret = String::with_capacity(v.len());
        for c in decode_utf16(v.iter().cloned()) {
            if let Ok(c) = c {
                ret.push(c);
            } else {
                return Err(Utf16Error);
            }
        }
        Ok(ret)
    }

    #[inline]
    pub fn retain<F>(&mut self, mut f: F)
        where
            F: FnMut(char) -> bool,
    {
        struct SetLenOnDrop<'a> {
            s: &'a mut String,
            idx: usize,
            del_bytes: usize,
        }

        impl<'a> Drop for SetLenOnDrop<'a> {
            fn drop(&mut self) {
                let new_len = self.idx - self.del_bytes;
                debug_assert!(new_len <= self.s.len());
                unsafe { self.s.vec.set_len(new_len) };
            }
        }

        let len = self.len();
        let mut guard = SetLenOnDrop { s: self, idx: 0, del_bytes: 0 };

        while guard.idx < len {
            let ch =
                // SAFETY: `guard.idx` is positive-or-zero and less that len so the `get_unchecked`
                // is in bound. `self` is valid UTF-8 like string and the returned slice starts at
                // a unicode code point so the `Chars` always return one character.
                unsafe { guard.s.get_unchecked(guard.idx..len).chars().next().unwrap_unchecked() };
            let ch_len = ch.len_utf8();

            if !f(ch) {
                guard.del_bytes += ch_len;
            } else if guard.del_bytes > 0 {
                // SAFETY: `guard.idx` is in bound and `guard.del_bytes` represent the number of
                // bytes that are erased from the string so the resulting `guard.idx -
                // guard.del_bytes` always represent a valid unicode code point.
                //
                // `guard.del_bytes` >= `ch.len_utf8()`, so taking a slice with `ch.len_utf8()` len
                // is safe.
                ch.encode_utf8(unsafe {
                    slice::from_raw_parts_mut(
                        guard.s.as_mut_ptr().add(guard.idx - guard.del_bytes),
                        ch.len_utf8(),
                    )
                });
            }

            // Point idx to the next char
            guard.idx += ch_len;
        }

        drop(guard);
    }
}

#[derive(Debug, Clone)]
pub struct Utf16Error;

impl Default for String {
    #[inline]
    fn default() -> Self {
        Self::new()
    }
}

impl AsRef<str> for String {
    fn as_ref(&self) -> &str {
        self
    }
}

impl AsMut<str> for String {
    fn as_mut(&mut self) -> &mut str {
        self
    }
}

impl core::ops::Deref for String {
    type Target = str;

    #[inline]
    fn deref(&self) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.vec) }
    }
}

impl core::ops::DerefMut for String {
    #[inline]
    fn deref_mut(&mut self) -> &mut str {
        unsafe { core::str::from_utf8_unchecked_mut(&mut self.vec) }
    }
}

impl core::borrow::Borrow<str> for String {
    fn borrow(&self) -> &str {
        self
    }
}

impl core::borrow::BorrowMut<str> for String {
    fn borrow_mut(&mut self) -> &mut str {
        self
    }
}

impl core::hash::Hash for String {
    fn hash<H: core::hash::Hasher>(&self, state: &mut H) {
        self.as_str().hash(state)
    }
}

impl core::str::FromStr for String {
    type Err = Infallible;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        Ok(String::from(s))
    }
}

impl Clone for String {
    fn clone(&self) -> Self {
        Self {
            vec: self.vec.clone(),
        }
    }
}

impl PartialEq for String {
    fn eq(&self, other: &Self) -> bool {
        self.vec == other.vec
    }
}

impl Eq for String {}

impl PartialOrd for String {
    fn partial_cmp(&self, other: &Self) -> Option<core::cmp::Ordering> {
        self.vec.partial_cmp(&other.vec)
    }
}

impl Ord for String {
    fn cmp(&self, other: &Self) -> core::cmp::Ordering {
        self.vec.cmp(&other.vec)
    }
}

impl core::fmt::Display for String {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&**self, f)
    }
}

impl core::fmt::Debug for String {
    #[inline]
    fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
        core::fmt::Display::fmt(&**self, f)
    }
}

impl From<char> for String {
    fn from(ch: char) -> Self {
        let mut bytes = unsafe { uninit_bits() };
        let bytes = ch.encode_utf8(&mut bytes).as_bytes();
        let vec = Vec::from(bytes);
        String { vec }
    }
}

impl From<&str> for String {
    fn from(string: &str) -> Self {
        let vec = Vec::from(string.as_bytes());
        String { vec }
    }
}

impl From<&String> for String {
    fn from(string: &String) -> Self {
        String::from(string.as_str())
    }
}

impl From<crate::boxed::Box<str>> for String {
    fn from(bx: crate::boxed::Box<str>) -> Self {
        String { vec: Vec::from(bx) }
    }
}

impl FromIterator<char> for String {
    fn from_iter<T: IntoIterator<Item = char>>(iter: T) -> Self {
        let mut s = String::new();
        s.extend(iter);
        s
    }
}

impl Extend<char> for String {
    fn extend<T: IntoIterator<Item = char>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for c in iter {
            self.push(c);
        }
    }
}

impl<'a> FromIterator<&'a str> for String {
    fn from_iter<T: IntoIterator<Item = &'a str>>(iter: T) -> Self {
        let mut s = String::new();
        s.extend(iter);
        s
    }
}

impl<'a> Extend<&'a str> for String {
    fn extend<T: IntoIterator<Item = &'a str>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        for s in iter {
            self.push_str(s);
        }
    }
}

impl<'a> FromIterator<&'a char> for String {
    fn from_iter<T: IntoIterator<Item = &'a char>>(iter: T) -> Self {
        let mut s = String::new();
        s.extend(iter);
        s
    }
}

impl<'a> Extend<&'a char> for String {
    fn extend<T: IntoIterator<Item = &'a char>>(&mut self, iter: T) {
        let iter = iter.into_iter();
        self.reserve(iter.size_hint().0);
        for c in iter {
            self.push(*c);
        }
    }
}

impl FromIterator<String> for String {
    fn from_iter<T: IntoIterator<Item = String>>(iter: T) -> Self {
        let mut s = String::new();
        s.extend(iter);
        s
    }
}

impl Extend<String> for String {
    fn extend<T: IntoIterator<Item = String>>(&mut self, iter: T) {
        let mut iter = iter.into_iter();

        // Use already allocated String if current one is not.
        if self.capacity() == 0 {
            if let Some(s) = iter.next() {
                self.vec = s.vec;
            }
        }

        for s in iter {
            self.push_str(&s);
        }
    }
}

impl FromIterator<crate::boxed::Box<str>> for String {
    fn from_iter<T: IntoIterator<Item = crate::boxed::Box<str>>>(iter: T) -> Self {
        let mut s = String::new();
        s.extend(iter);
        s
    }
}

impl Extend<crate::boxed::Box<str>> for String {
    fn extend<T: IntoIterator<Item = crate::boxed::Box<str>>>(&mut self, iter: T) {
        let iter = iter.into_iter().map(|bx| -> String { bx.into() });
        self.extend(iter);
    }
}

impl core::ops::Index<core::ops::Range<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: core::ops::Range<usize>) -> &str {
        &self[..][index]
    }
}

impl core::ops::Index<core::ops::RangeTo<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: core::ops::RangeTo<usize>) -> &str {
        &self[..][index]
    }
}

impl core::ops::Index<core::ops::RangeFrom<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: core::ops::RangeFrom<usize>) -> &str {
        &self[..][index]
    }
}

impl core::ops::Index<core::ops::RangeFull> for String {
    type Output = str;

    #[inline]
    fn index(&self, _index: core::ops::RangeFull) -> &str {
        unsafe { core::str::from_utf8_unchecked(&self.vec) }
    }
}

impl core::ops::Index<core::ops::RangeInclusive<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: core::ops::RangeInclusive<usize>) -> &str {
        core::ops::Index::index(&**self, index)
    }
}

impl core::ops::Index<core::ops::RangeToInclusive<usize>> for String {
    type Output = str;

    #[inline]
    fn index(&self, index: core::ops::RangeToInclusive<usize>) -> &str {
        core::ops::Index::index(&**self, index)
    }
}

impl core::ops::IndexMut<core::ops::Range<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: core::ops::Range<usize>) -> &mut str {
        &mut self[..][index]
    }
}

impl core::ops::IndexMut<core::ops::RangeTo<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: core::ops::RangeTo<usize>) -> &mut str {
        &mut self[..][index]
    }
}

impl core::ops::IndexMut<core::ops::RangeFrom<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: core::ops::RangeFrom<usize>) -> &mut str {
        &mut self[..][index]
    }
}

impl core::ops::IndexMut<core::ops::RangeFull> for String {
    #[inline]
    fn index_mut(&mut self, _index: core::ops::RangeFull) -> &mut str {
        unsafe { core::str::from_utf8_unchecked_mut(&mut *self.vec) }
    }
}

impl core::ops::IndexMut<core::ops::RangeInclusive<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: core::ops::RangeInclusive<usize>) -> &mut str {
        core::ops::IndexMut::index_mut(&mut **self, index)
    }
}

impl core::ops::IndexMut<core::ops::RangeToInclusive<usize>> for String {
    #[inline]
    fn index_mut(&mut self, index: core::ops::RangeToInclusive<usize>) -> &mut str {
        core::ops::IndexMut::index_mut(&mut **self, index)
    }
}

impl core::ops::Add<&str> for String {
    type Output = String;

    #[inline]
    fn add(mut self, other: &str) -> String {
        self.push_str(other);
        self
    }
}

impl core::ops::AddAssign<&str> for String {
    #[inline]
    fn add_assign(&mut self, other: &str) {
        self.push_str(other);
    }
}

impl core::fmt::Write for String {
    #[inline]
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        self.push_str(s);
        Ok(())
    }

    #[inline]
    fn write_char(&mut self, c: char) -> core::fmt::Result {
        self.push(c);
        Ok(())
    }
}

unsafe fn uninit_bits() -> [u8; 4] {
    MaybeUninit::<[MaybeUninit<u8>; 4]>::uninit()
        .assume_init()
        .map(|v| v.assume_init())
}
