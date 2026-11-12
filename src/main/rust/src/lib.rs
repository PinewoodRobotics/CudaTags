pub mod bridge;

#[cfg(test)]
mod tests {
    use super::bridge::ffi::*;

    #[test]
    fn image_borrows_pixels_without_native_backend() {
        let mut pixels = [0u8; 12];
        let image = RustImage {
            pixels: &mut pixels,
            width: 2,
            height: 2,
            stride: 6,
            channels: 3,
        };
        image.pixels[0] = 42;
        assert_eq!(pixels[0], 42);
    }
}
