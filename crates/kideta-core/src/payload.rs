//! Payload types for vector metadata.
//!
//! This module provides types for storing arbitrary metadata alongside vectors.
//! Payloads are key-value stores where values can be strings, numbers, booleans,
//! geographic locations, or nested lists.
//!
//! # PayloadValue
//!
//! [`PayloadValue`] is an enum that can hold different types of values:
//!
//! - [`Str`](PayloadValue::Str) - UTF-8 string
//! - [`Int`](PayloadValue::Int) - 64-bit signed integer
//! - [`Float`](PayloadValue::Float) - 64-bit floating point
//! - [`Bool`](PayloadValue::Bool) - Boolean
//! - [`Geo`](PayloadValue::Geo) - Geographic point (latitude, longitude)
//! - [`List`](PayloadValue::List) - Nested list of payload values
//!
//! # Payload
//!
//! [`Payload`] is a HashMap from string keys to [`PayloadValue`] entries.
//! It provides type-safe access to values through helper methods.
//!
//! # Example
//!
//! ```
//! use kideta_core::payload::{Payload, PayloadValue, GeoPoint};
//!
//! // Create a payload with various field types
//! let mut payload = Payload::new();
//! payload.insert("title", PayloadValue::Str("Hello World".to_string()));
//! payload.insert("views", PayloadValue::Int(42));
//! payload.insert("rating", PayloadValue::Float(4.5));
//! payload.insert("active", PayloadValue::Bool(true));
//! payload.insert("location", PayloadValue::Geo(GeoPoint { lat: 37.7749, lon: -122.4194 }));
//! payload.insert("tags", PayloadValue::List(vec![
//!     PayloadValue::Str("rust".to_string()),
//!     PayloadValue::Str("vector".to_string()),
//! ]));
//!
//! // Access values with type-safe getters
//! assert_eq!(payload.get("title").and_then(|v| v.as_str()), Some("Hello World"));
//! assert_eq!(payload.get("views").and_then(|v| v.as_int()), Some(42));
//! assert_eq!(payload.get("rating").and_then(|v| v.as_float()), Some(4.5));
//! assert_eq!(payload.get("active").and_then(|v| v.as_bool()), Some(true));
//!
//! // Geographic point access
//! let geo = payload.get("location").and_then(|v| v.as_geo()).unwrap();
//! assert!((geo.lat - 37.7749).abs() < 0.0001);
//! ```
//!
//! # Serialization
//!
//! Both `PayloadValue` and `Payload` serialize to/from JSON:
//!
//! ```rust
//! use kideta_core::payload::{Payload, PayloadValue};
//! use std::collections::HashMap;
//!
//! let mut map = HashMap::new();
//! map.insert("name".to_string(), PayloadValue::Str("test".to_string()));
//! map.insert("count".to_string(), PayloadValue::Int(5));
//! let payload = Payload::from(map);
//!
//! let json = serde_json::to_string(&payload).unwrap();
//! let parsed: Payload = serde_json::from_str(&json).unwrap();
//! assert_eq!(parsed, payload);
//! ```

use serde::{Deserialize, Serialize};
use std::cmp::Ordering;
use std::collections::HashMap;

/// A geographic point with latitude and longitude coordinates.
///
/// Used for geo-spatial filtering and queries based on location.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GeoPoint {
    /// Latitude in degrees (must be in range [-90, 90]).
    pub lat: f64,

    /// Longitude in degrees (must be in range [-180, 180]).
    pub lon: f64,
}

impl PartialOrd for GeoPoint {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        // Compare by latitude first, then longitude
        match self.lat.partial_cmp(&other.lat) {
            Some(Ordering::Equal) => self.lon.partial_cmp(&other.lon),
            other => other,
        }
    }
}

/// A typed payload value that can be stored in a vector's metadata.
///
/// See the [module-level documentation](self) for more information and examples.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PayloadValue {
    /// UTF-8 encoded string value.
    Str(String),

    /// 64-bit signed integer value.
    Int(i64),

    /// 64-bit floating point value.
    Float(f64),

    /// Boolean value.
    Bool(bool),

    /// Geographic point (latitude, longitude).
    Geo(GeoPoint),

    /// Nested list of payload values.
    List(Vec<PayloadValue>),
}

impl PartialOrd for PayloadValue {
    fn partial_cmp(
        &self,
        other: &Self,
    ) -> Option<Ordering> {
        match (self, other) {
            // Same type comparisons
            (PayloadValue::Int(a), PayloadValue::Int(b)) => a.partial_cmp(b),
            (PayloadValue::Float(a), PayloadValue::Float(b)) => a.partial_cmp(b),
            (PayloadValue::Str(a), PayloadValue::Str(b)) => a.partial_cmp(b),
            (PayloadValue::Bool(a), PayloadValue::Bool(b)) => a.partial_cmp(b),
            (PayloadValue::Geo(a), PayloadValue::Geo(b)) => a.partial_cmp(b),

            // Cross-type numeric comparisons (Int vs Float)
            (PayloadValue::Int(a), PayloadValue::Float(b)) => (*a as f64).partial_cmp(b),
            (PayloadValue::Float(a), PayloadValue::Int(b)) => a.partial_cmp(&(*b as f64)),

            // Lists are not comparable
            (PayloadValue::List(_), _) | (_, PayloadValue::List(_)) => None,

            // Different types are not comparable
            _ => None,
        }
    }
}

impl PayloadValue {
    /// Returns true if this value can be compared with another.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::PayloadValue;
    ///
    /// let int_val = PayloadValue::Int(42);
    /// let float_val = PayloadValue::Float(42.0);
    /// let str_val = PayloadValue::Str("test".to_string());
    ///
    /// assert!(int_val.is_comparable_with(&float_val));
    /// assert!(int_val.is_comparable_with(&PayloadValue::Int(10)));
    /// assert!(!int_val.is_comparable_with(&str_val));
    /// ```
    pub fn is_comparable_with(
        &self,
        other: &Self,
    ) -> bool {
        matches!(
            (self, other),
            (PayloadValue::Int(_), PayloadValue::Int(_))
                | (PayloadValue::Float(_), PayloadValue::Float(_))
                | (PayloadValue::Int(_), PayloadValue::Float(_))
                | (PayloadValue::Float(_), PayloadValue::Int(_))
                | (PayloadValue::Str(_), PayloadValue::Str(_))
                | (PayloadValue::Bool(_), PayloadValue::Bool(_))
                | (PayloadValue::Geo(_), PayloadValue::Geo(_))
        )
    }

    /// Returns the string value if `self` is a `Str` variant.
    /// Returns the string value if `self` is a `Str` variant.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::PayloadValue;
    ///
    /// let value = PayloadValue::Str("hello".to_string());
    /// assert_eq!(value.as_str(), Some("hello"));
    ///
    /// let not_string = PayloadValue::Int(42);
    /// assert_eq!(not_string.as_str(), None);
    /// ```
    pub fn as_str(&self) -> Option<&str> {
        match self {
            PayloadValue::Str(s) => Some(s),
            _ => None,
        }
    }

    /// Returns the integer value if `self` is an `Int` variant.
    pub fn as_int(&self) -> Option<i64> {
        match self {
            PayloadValue::Int(i) => Some(*i),
            _ => None,
        }
    }

    /// Returns the float value if `self` is a `Float` variant.
    pub fn as_float(&self) -> Option<f64> {
        match self {
            PayloadValue::Float(f) => Some(*f),
            _ => None,
        }
    }

    /// Returns the boolean value if `self` is a `Bool` variant.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            PayloadValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Returns the geo point if `self` is a `Geo` variant.
    pub fn as_geo(&self) -> Option<&GeoPoint> {
        match self {
            PayloadValue::Geo(g) => Some(g),
            _ => None,
        }
    }

    /// Returns the list value if `self` is a `List` variant.
    pub fn as_list(&self) -> Option<&[PayloadValue]> {
        match self {
            PayloadValue::List(v) => Some(v),
            _ => None,
        }
    }
}

/// A collection of key-value metadata associated with a vector.
///
/// `Payload` is essentially a `HashMap<String, PayloadValue>` with
/// type-safe accessors and helper methods.
///
/// # Creation
///
/// ```
/// use kideta_core::payload::{Payload, PayloadValue};
///
/// // Empty payload
/// let mut payload = Payload::new();
///
/// // With initial capacity hint
/// let payload = Payload::with_capacity(10);
///
/// // From a HashMap
/// use std::collections::HashMap;
/// let mut map = HashMap::new();
/// map.insert("key".to_string(), PayloadValue::Int(42));
/// let payload = Payload::from(map);
/// ```
///
/// # Operations
///
/// ```
/// use kideta_core::payload::{Payload, PayloadValue};
///
/// let mut payload = Payload::new();
///
/// // Insert values
/// payload.insert("name", PayloadValue::Str("test".to_string()));
///
/// // Check existence
/// assert!(payload.contains_key("name"));
/// assert!(!payload.contains_key("missing"));
///
/// // Get values
/// assert!(payload.get("name").is_some());
///
/// // Remove values
/// payload.remove("name");
/// assert!(!payload.contains_key("name"));
/// ```
///
/// # Iteration
///
/// ```
/// use kideta_core::payload::{Payload, PayloadValue};
///
/// let mut payload = Payload::new();
/// payload.insert("a", PayloadValue::Int(1));
/// payload.insert("b", PayloadValue::Int(2));
///
/// for (key, value) in payload.iter() {
///     println!("{}: {:?}", key, value);
/// }
/// ```
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Payload(HashMap<String, PayloadValue>);

impl Payload {
    /// Creates a new empty `Payload`.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::Payload;
    /// let payload = Payload::new();
    /// assert!(payload.is_empty());
    /// ```
    pub fn new() -> Self {
        Payload(HashMap::new())
    }

    /// Creates a new empty `Payload` with at least the specified capacity.
    ///
    /// This is useful when you know the approximate number of fields
    /// upfront to avoid reallocations.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::Payload;
    /// let payload = Payload::with_capacity(100);
    /// assert!(payload.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        Payload(HashMap::with_capacity(capacity))
    }

    /// Inserts a key-value pair into the payload.
    ///
    /// Returns the old value if the key was already present.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::{Payload, PayloadValue};
    ///
    /// let mut payload = Payload::new();
    /// let old = payload.insert("count", PayloadValue::Int(1));
    /// assert!(old.is_none());
    ///
    /// // Replace existing value
    /// let old = payload.insert("count", PayloadValue::Int(2));
    /// assert!(matches!(old, Some(PayloadValue::Int(1))));
    /// ```
    pub fn insert(
        &mut self,
        key: impl Into<String>,
        value: PayloadValue,
    ) -> Option<PayloadValue> {
        self.0.insert(key.into(), value)
    }

    /// Gets a reference to the value for the given key.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::{Payload, PayloadValue};
    ///
    /// let mut payload = Payload::new();
    /// payload.insert("key", PayloadValue::Int(42));
    ///
    /// assert_eq!(payload.get("key").and_then(|v| v.as_int()), Some(42));
    /// assert!(payload.get("missing").is_none());
    /// ```
    pub fn get(
        &self,
        key: &str,
    ) -> Option<&PayloadValue> {
        self.0.get(key)
    }

    /// Removes a key from the payload, returning the value if it existed.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::payload::{Payload, PayloadValue};
    ///
    /// let mut payload = Payload::new();
    /// payload.insert("key", PayloadValue::Int(42));
    ///
    /// let removed = payload.remove("key");
    /// assert!(matches!(removed, Some(PayloadValue::Int(42))));
    /// assert!(!payload.contains_key("key"));
    /// ```
    pub fn remove(
        &mut self,
        key: &str,
    ) -> Option<PayloadValue> {
        self.0.remove(key)
    }

    /// Returns `true` if the payload contains the given key.
    pub fn contains_key(
        &self,
        key: &str,
    ) -> bool {
        self.0.contains_key(key)
    }

    /// Returns the number of key-value pairs in the payload.
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Returns `true` if the payload contains no key-value pairs.
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// Returns an iterator over all key-value pairs in the payload.
    pub fn iter(&self) -> impl Iterator<Item = (&str, &PayloadValue)> {
        self.0.iter().map(|(k, v)| (k.as_str(), v))
    }
}

impl From<HashMap<String, PayloadValue>> for Payload {
    fn from(map: HashMap<String, PayloadValue>) -> Self {
        Payload(map)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_payload_basic_operations() {
        let mut payload = Payload::new();
        payload.insert("name", PayloadValue::Str("test".to_string()));
        payload.insert("age", PayloadValue::Int(25));
        payload.insert("score", PayloadValue::Float(3.14));

        assert_eq!(payload.len(), 3);
        assert_eq!(payload.get("name").and_then(|v| v.as_str()), Some("test"));
        assert_eq!(payload.get("age").and_then(|v| v.as_int()), Some(25));
        assert_eq!(payload.get("score").and_then(|v| v.as_float()), Some(3.14));
    }

    #[test]
    fn test_payload_geo() {
        let mut payload = Payload::new();
        payload.insert(
            "location",
            PayloadValue::Geo(GeoPoint {
                lat: 37.7749,
                lon: -122.4194,
            }),
        );
        let geo = payload
            .get("location")
            .and_then(|v| v.as_geo())
            .unwrap();
        assert!((geo.lat - 37.7749).abs() < 1e-6);
    }

    #[test]
    fn test_payload_list() {
        let mut payload = Payload::new();
        payload.insert(
            "tags",
            PayloadValue::List(vec![
                PayloadValue::Str("rust".to_string()),
                PayloadValue::Str("vector".to_string()),
            ]),
        );
        let list = payload
            .get("tags")
            .and_then(|v| v.as_list())
            .unwrap();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_payload_remove() {
        let mut payload = Payload::new();
        payload.insert("key", PayloadValue::Int(42));
        assert!(payload.contains_key("key"));
        payload.remove("key");
        assert!(!payload.contains_key("key"));
    }

    #[test]
    fn test_payload_serde() {
        let mut payload = Payload::new();
        payload.insert("name", PayloadValue::Str("test".to_string()));
        payload.insert("value", PayloadValue::Int(100));

        let serialized = serde_json::to_string(&payload).unwrap();
        let deserialized: Payload = serde_json::from_str(&serialized).unwrap();
        assert_eq!(deserialized, payload);
    }

    #[test]
    fn test_payload_value_serde() {
        let value = PayloadValue::Str("hello".to_string());
        let json = serde_json::to_string(&value).unwrap();
        let parsed: PayloadValue = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, value);
    }

    #[test]
    fn test_geopoint_comparison() {
        let point1 = GeoPoint {
            lat: 37.0,
            lon: -122.0,
        };
        let point2 = GeoPoint {
            lat: 37.0,
            lon: -121.0,
        };
        let point3 = GeoPoint {
            lat: 38.0,
            lon: -122.0,
        };

        assert!(point1 < point2); // Same lat, smaller lon
        assert!(point1 < point3); // Smaller lat
        assert!(point2 < point3); // point2 has smaller lat
    }

    #[test]
    fn test_payloadvalue_comparison() {
        // Int comparisons
        assert!(PayloadValue::Int(10) < PayloadValue::Int(20));
        assert!(PayloadValue::Int(20) > PayloadValue::Int(10));
        assert_eq!(
            PayloadValue::Int(10).partial_cmp(&PayloadValue::Int(10)),
            Some(Ordering::Equal)
        );

        // Float comparisons
        assert!(PayloadValue::Float(10.5) < PayloadValue::Float(20.5));

        // String comparisons
        assert!(PayloadValue::Str("a".to_string()) < PayloadValue::Str("b".to_string()));

        // Cross-type numeric (Int vs Float)
        assert!(PayloadValue::Int(10) < PayloadValue::Float(20.0));
        assert!(PayloadValue::Float(10.0) < PayloadValue::Int(20));

        // Incomparable types
        assert_eq!(
            PayloadValue::Int(10).partial_cmp(&PayloadValue::Str("10".to_string())),
            None
        );

        // Lists are not comparable
        assert_eq!(
            PayloadValue::List(vec![]).partial_cmp(&PayloadValue::List(vec![])),
            None
        );
    }

    #[test]
    fn test_is_comparable_with() {
        let int_val = PayloadValue::Int(42);
        let float_val = PayloadValue::Float(42.0);
        let str_val = PayloadValue::Str("test".to_string());

        assert!(int_val.is_comparable_with(&float_val));
        assert!(int_val.is_comparable_with(&PayloadValue::Int(10)));
        assert!(float_val.is_comparable_with(&PayloadValue::Float(10.0)));
        assert!(!int_val.is_comparable_with(&str_val));
        assert!(!str_val.is_comparable_with(&int_val));

        // Test geo comparison
        let geo1 = PayloadValue::Geo(GeoPoint {
            lat: 37.0,
            lon: -122.0,
        });
        let geo2 = PayloadValue::Geo(GeoPoint {
            lat: 38.0,
            lon: -122.0,
        });
        assert!(geo1.is_comparable_with(&geo2));
    }
}
