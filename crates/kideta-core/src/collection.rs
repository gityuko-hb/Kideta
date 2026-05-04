//! Collection configuration and schema definitions.
//!
//! Collections are the top-level organizational unit in Kideta. Each collection
//! has a schema that defines its vector fields and scalar fields, along with
//! validation rules.
//!
//! # CollectionConfig
//!
//! [`CollectionConfig`] defines the structure of a collection:
//!
//! - Vector fields (embeddings for similarity search)
//! - Scalar fields (metadata for filtering)
//!
//! # CollectionSchema
//!
//! [`CollectionSchema`] wraps a [`CollectionConfig`] with additional
//! validation logic. Before creating a collection, you should validate
//! the schema to catch configuration errors early.
//!
//! # Example: Creating a Collection
//!
//! ```
//! use kideta_core::collection::{CollectionConfig, CollectionSchema};
//! use kideta_core::schema::{ScalarField, ScalarFieldType, VectorField};
//! use kideta_core::metric::DistanceMetric;
//!
//! // Define the collection configuration
//! let config = CollectionConfig::new("my_collection")
//!     .add_vector_field(
//!         VectorField::new("text_embedding", 768, DistanceMetric::Cosine)
//!     )
//!     .add_scalar_field(
//!         ScalarField::new("category", ScalarFieldType::Str).indexed(true)
//!     )
//!     .add_scalar_field(
//!         ScalarField::new("price", ScalarFieldType::Float).indexed(true)
//!     );
//!
//! // Wrap in a schema for validation
//! let schema = CollectionSchema::from_config(config);
//!
//! // Validate before creating the collection
//! schema.validate().expect("invalid schema");
//! ```
//!
//! # Validation Rules
//!
//! [`CollectionSchema::validate`] checks:
//!
//! - At least one vector field must be defined
//! - Vector field names must be unique
//! - Scalar field names must be unique
//! - No vector field can have dimension 0

use crate::schema::{ScalarField, VectorField};
use serde::{Deserialize, Serialize};

/// Configuration for a collection.
///
/// This defines the structure of a collection but does not include
/// validation. Use [`CollectionSchema`] if you need validation.
///
/// # Creating a Collection
///
/// ```
/// use kideta_core::collection::CollectionConfig;
/// use kideta_core::schema::{ScalarField, ScalarFieldType, VectorField};
/// use kideta_core::metric::DistanceMetric;
///
/// let config = CollectionConfig::new("my_vectors")
///     .add_vector_field(
///         VectorField::new("embedding", 768, DistanceMetric::Cosine)
///     );
/// ```
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionConfig {
    /// Unique name of this collection.
    pub name: String,

    /// Vector fields (embeddings) for this collection.
    pub vectors: Vec<VectorField>,

    /// Scalar fields (metadata) for this collection.
    pub scalar_fields: Vec<ScalarField>,
}

impl CollectionConfig {
    /// Creates a new empty collection configuration with the given name.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    ///
    /// let config = CollectionConfig::new("products");
    /// assert_eq!(config.name, "products");
    /// assert!(config.vectors.is_empty());
    /// assert!(config.scalar_fields.is_empty());
    /// ```
    pub fn new(name: impl Into<String>) -> Self {
        CollectionConfig {
            name: name.into(),
            vectors: Vec::new(),
            scalar_fields: Vec::new(),
        }
    }

    /// Adds a vector field to this collection.
    ///
    /// Returns a new `CollectionConfig` with the field added.
    /// This allows for a fluent builder pattern.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let config = CollectionConfig::new("my_collection")
    ///     .add_vector_field(
    ///         VectorField::new("embedding", 768, DistanceMetric::Cosine)
    ///     );
    ///
    /// assert_eq!(config.vectors.len(), 1);
    /// ```
    pub fn add_vector_field(
        mut self,
        field: VectorField,
    ) -> Self {
        self.vectors.push(field);
        self
    }

    /// Adds a scalar field to this collection.
    ///
    /// Returns a new `CollectionConfig` with the field added.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    /// use kideta_core::schema::{ScalarField, ScalarFieldType};
    ///
    /// let config = CollectionConfig::new("my_collection")
    ///     .add_scalar_field(
    ///         ScalarField::new("category", ScalarFieldType::Str).indexed(true)
    ///     );
    ///
    /// assert_eq!(config.scalar_fields.len(), 1);
    /// ```
    pub fn add_scalar_field(
        mut self,
        field: ScalarField,
    ) -> Self {
        self.scalar_fields.push(field);
        self
    }

    /// Gets a vector field by name.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let config = CollectionConfig::new("my_collection")
    ///     .add_vector_field(
    ///         VectorField::new("embedding", 768, DistanceMetric::Cosine)
    ///     );
    ///
    /// let field = config.get_vector_field("embedding");
    /// assert!(field.is_some());
    ///
    /// let missing = config.get_vector_field("nonexistent");
    /// assert!(missing.is_none());
    /// ```
    pub fn get_vector_field(
        &self,
        name: &str,
    ) -> Option<&VectorField> {
        self.vectors.iter().find(|f| f.name == name)
    }

    /// Gets a scalar field by name.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    /// use kideta_core::schema::{ScalarField, ScalarFieldType};
    ///
    /// let config = CollectionConfig::new("my_collection")
    ///     .add_scalar_field(
    ///         ScalarField::new("category", ScalarFieldType::Str)
    ///     );
    ///
    /// let field = config.get_scalar_field("category");
    /// assert!(field.is_some());
    /// ```
    pub fn get_scalar_field(
        &self,
        name: &str,
    ) -> Option<&ScalarField> {
        self.scalar_fields.iter().find(|f| f.name == name)
    }

    /// Returns the sum of all vector field dimensions.
    ///
    /// This is useful for calculating total memory requirements
    /// or validating record consistency.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionConfig;
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let config = CollectionConfig::new("multi_vector")
    ///     .add_vector_field(VectorField::new("text", 768, DistanceMetric::Cosine))
    ///     .add_vector_field(VectorField::new("image", 512, DistanceMetric::L2));
    ///
    /// assert_eq!(config.total_vector_dimensions(), 1280);
    /// ```
    pub fn total_vector_dimensions(&self) -> usize {
        self.vectors.iter().map(|f| f.dimension).sum()
    }

    /// Returns the name of the first vector field, or None if no vectors defined.
    pub fn default_vector_field(&self) -> Option<&str> {
        self.vectors.first().map(|f| f.name.as_str())
    }
}

/// A validated collection schema.
///
/// [`CollectionSchema`] wraps a [`CollectionConfig`] and provides
/// validation to ensure the schema is well-formed before creating
/// a collection.
///
/// # Validation
///
/// Use [`validate()`](CollectionSchema::validate) to check if a schema
/// is valid before creating a collection:
///
/// ```
/// use kideta_core::collection::{CollectionConfig, CollectionSchema};
/// use kideta_core::schema::VectorField;
/// use kideta_core::metric::DistanceMetric;
///
/// let schema = CollectionSchema::from_config(
///     CollectionConfig::new("test")
///         .add_vector_field(VectorField::new("embedding", 768, DistanceMetric::Cosine))
/// );
///
/// assert!(schema.validate().is_ok());
/// ```
///
/// # Invalid Schemas
///
/// Validation will fail for schemas that:
///
/// - Have no vector fields
/// - Have duplicate vector field names
/// - Have duplicate scalar field names
/// - Have vector fields with dimension 0
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CollectionSchema {
    /// The underlying collection configuration.
    pub config: CollectionConfig,
}

impl CollectionSchema {
    /// Creates a new schema from a validated configuration.
    ///
    /// Note: This does not validate the configuration. Use
    /// [`validate()`](CollectionSchema::validate) to validate
    /// before creating a collection.
    pub fn from_config(config: CollectionConfig) -> Self {
        CollectionSchema { config }
    }

    /// Validates this schema.
    ///
    /// Returns `Ok(())` if the schema is valid, or an `Err` with
    /// a description of the validation error.
    ///
    /// # Validation Rules
    ///
    /// - At least one vector field must be defined
    /// - Vector field names must be unique within the collection
    /// - Scalar field names must be unique within the collection
    /// - Vector fields must have dimension > 0
    ///
    /// # Example: Valid Schema
    ///
    /// ```
    /// use kideta_core::collection::{CollectionConfig, CollectionSchema};
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let schema = CollectionSchema::from_config(
    ///     CollectionConfig::new("test")
    ///         .add_vector_field(VectorField::new("embedding", 768, DistanceMetric::Cosine))
    /// );
    ///
    /// assert!(schema.validate().is_ok());
    /// ```
    ///
    /// # Example: Invalid Schema (Empty)
    ///
    /// ```
    /// use kideta_core::collection::{CollectionConfig, CollectionSchema};
    ///
    /// let schema = CollectionSchema::from_config(CollectionConfig::new("empty"));
    /// assert!(schema.validate().is_err());
    /// ```
    ///
    /// # Example: Invalid Schema (Duplicate Names)
    ///
    /// ```
    /// use kideta_core::collection::{CollectionConfig, CollectionSchema};
    /// use kideta_core::schema::VectorField;
    /// use kideta_core::metric::DistanceMetric;
    ///
    /// let schema = CollectionSchema::from_config(
    ///     CollectionConfig::new("test")
    ///         .add_vector_field(VectorField::new("embedding", 768, DistanceMetric::Cosine))
    ///         .add_vector_field(VectorField::new("embedding", 384, DistanceMetric::L2)) // Duplicate!
    /// );
    ///
    /// assert!(schema.validate().is_err());
    /// ```
    pub fn validate(&self) -> Result<(), String> {
        if self.config.vectors.is_empty() {
            return Err("Collection must have at least one vector field".to_string());
        }

        let mut vector_names = std::collections::HashSet::new();
        for field in &self.config.vectors {
            if !vector_names.insert(&field.name) {
                return Err(format!("Duplicate vector field name: {}", field.name));
            }
            if field.dimension == 0 {
                return Err(format!(
                    "Vector field '{}' has invalid dimension 0",
                    field.name
                ));
            }
        }

        let mut scalar_names = std::collections::HashSet::new();
        for field in &self.config.scalar_fields {
            if !scalar_names.insert(&field.name) {
                return Err(format!("Duplicate scalar field name: {}", field.name));
            }
        }

        Ok(())
    }
}

/// A collection alias — an indirection from a human-readable name to a collection name.
///
/// Aliases enable zero-downtime collection swaps for blue-green deployments.
/// When you need to rebuild a collection's index, you can:
///
/// 1. Create the new collection with a temporary name (e.g., `products_v2`)
/// 2. Point the alias `products` to the new collection
/// 3. Delete the old collection
///
/// Clients always connect via the alias, so they experience no downtime.
///
/// # Example
///
/// ```
/// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
///
/// let mut map = CollectionAliasMap::new();
///
/// // Create an alias pointing to v1
/// let alias = CollectionAlias::new("my_products").unwrap();
/// map.set(alias.clone(), "products_v1".to_string());
///
/// // Resolve the alias
/// assert_eq!(map.get(&alias), Some(&"products_v1".to_string()));
///
/// // Atomic swap to v2 (blue-green deployment)
/// let old = map.swap(&alias, "products_v2".to_string());
/// assert_eq!(old, Some("products_v1".to_string()));
/// assert_eq!(map.get(&alias), Some(&"products_v2".to_string()));
/// ```
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct CollectionAlias(String);

impl CollectionAlias {
    /// Creates a new `CollectionAlias` from a string.
    ///
    /// An alias name must be non-empty. Alias names are used by clients
    /// to reference collections without hard-coding collection names.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionAlias;
    ///
    /// let alias = CollectionAlias::new("my_collection").unwrap();
    /// assert_eq!(alias.as_str(), "my_collection");
    ///
    /// // Empty alias is rejected
    /// assert!(CollectionAlias::new("").is_err());
    /// ```
    pub fn new(name: impl Into<String>) -> Result<Self, String> {
        let name = name.into();
        if name.is_empty() {
            Err("Alias name cannot be empty".to_string())
        } else {
            Ok(CollectionAlias(name))
        }
    }

    /// Returns the underlying alias name as a string slice.
    #[inline]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for CollectionAlias {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// A map from collection aliases to collection names.
///
/// [`CollectionAliasMap`] manages the alias-to-collection-name indirection layer.
/// It supports atomic swaps, which are essential for zero-downtime blue-green
/// deployments where you need to switch an alias from an old collection to a new one.
///
/// # Example: Blue-Green Deployment
///
/// ```
/// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
///
/// let mut map = CollectionAliasMap::new();
///
/// // Setup: "products" alias points to v1
/// let alias = CollectionAlias::new("products").unwrap();
/// map.set(alias.clone(), "products_v1".to_string());
///
/// // Action: atomically switch to v2
/// let previous = map.swap(&alias, "products_v2".to_string());
///
/// assert_eq!(previous, Some("products_v1".to_string()));
/// assert_eq!(map.get(&alias), Some(&"products_v2".to_string()));
/// ```
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct CollectionAliasMap {
    inner: std::collections::HashMap<CollectionAlias, String>,
}

impl CollectionAliasMap {
    /// Creates a new empty alias map.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionAliasMap;
    ///
    /// let map = CollectionAliasMap::new();
    /// assert!(map.is_empty());
    /// ```
    pub fn new() -> Self {
        CollectionAliasMap {
            inner: std::collections::HashMap::new(),
        }
    }

    /// Creates an alias map with the given capacity.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::CollectionAliasMap;
    ///
    /// let map = CollectionAliasMap::with_capacity(100);
    /// assert!(map.is_empty());
    /// ```
    pub fn with_capacity(capacity: usize) -> Self {
        CollectionAliasMap {
            inner: std::collections::HashMap::with_capacity(capacity),
        }
    }

    /// Returns the number of aliases in this map.
    #[inline]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Returns `true` if this map contains no aliases.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.inner.is_empty()
    }

    /// Returns the collection name that the given alias points to.
    ///
    /// Returns `None` if the alias does not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// let alias = CollectionAlias::new("app").unwrap();
    /// map.set(alias.clone(), "production_v3".to_string());
    ///
    /// assert_eq!(map.get(&alias), Some(&"production_v3".to_string()));
    /// assert_eq!(map.get(&CollectionAlias::new("nonexistent").unwrap()), None);
    /// ```
    pub fn get(
        &self,
        alias: &CollectionAlias,
    ) -> Option<&String> {
        self.inner.get(alias)
    }

    /// Sets the target of an alias, replacing any existing target.
    ///
    /// Returns the previous collection name if the alias existed.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// let alias = CollectionAlias::new("search").unwrap();
    ///
    /// // First set
    /// let prev = map.set(alias.clone(), "search_v1".to_string());
    /// assert_eq!(prev, None);
    ///
    /// // Replace
    /// let prev = map.set(alias.clone(), "search_v2".to_string());
    /// assert_eq!(prev, Some("search_v1".to_string()));
    /// ```
    pub fn set(
        &mut self,
        alias: CollectionAlias,
        target: String,
    ) -> Option<String> {
        self.inner.insert(alias, target)
    }

    /// Removes an alias from the map.
    ///
    /// Returns the collection name that the alias pointed to, or `None`
    /// if the alias did not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// let alias = CollectionAlias::new("temp").unwrap();
    /// map.set(alias.clone(), "temp_collection".to_string());
    ///
    /// let removed = map.remove(&alias);
    /// assert_eq!(removed, Some("temp_collection".to_string()));
    /// assert!(map.get(&alias).is_none());
    /// ```
    pub fn remove(
        &mut self,
        alias: &CollectionAlias,
    ) -> Option<String> {
        self.inner.remove(alias)
    }

    /// Atomically swaps the target of an alias from one collection to another.
    ///
    /// This is the core operation for blue-green deployments:
    ///
    /// 1. You have `collection_v1` and `collection_v2` (the new version)
    /// 2. Your alias `collection` points to `collection_v1`
    /// 3. When v2 is ready, call `swap(alias, "collection_v2")`
    /// 4. The alias now points to v2, and clients using the alias see the new version
    ///
    /// Returns the previous collection name that the alias pointed to,
    /// or `None` if the alias did not exist.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// let alias = CollectionAlias::new("products").unwrap();
    /// map.set(alias.clone(), "products_blue".to_string());
    ///
    /// // Atomic swap to the green version
    /// let old = map.swap(&alias, "products_green".to_string());
    ///
    /// assert_eq!(old, Some("products_blue".to_string()));
    /// assert_eq!(map.get(&alias), Some(&"products_green".to_string()));
    /// ```
    pub fn swap(
        &mut self,
        alias: &CollectionAlias,
        new_target: String,
    ) -> Option<String> {
        self.inner.insert(alias.clone(), new_target)
    }

    /// Returns an iterator over all alias-target pairs.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// map.set(CollectionAlias::new("a").unwrap(), "col_a".to_string());
    /// map.set(CollectionAlias::new("b").unwrap(), "col_b".to_string());
    ///
    /// let mut iter = map.iter();
    /// let (alias, target) = iter.next().unwrap();
    /// // HashMap iteration order is not guaranteed, so check which key we got
    /// match alias.as_str() {
    ///     "a" => assert_eq!(target, "col_a"),
    ///     "b" => assert_eq!(target, "col_b"),
    ///     _ => panic!("unexpected alias"),
    /// }
    /// ```
    pub fn iter(&self) -> impl Iterator<Item = (&CollectionAlias, &String)> {
        self.inner.iter()
    }

    /// Returns `true` if this map contains the given alias.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// let alias = CollectionAlias::new("main").unwrap();
    /// map.set(alias.clone(), "main_v1".to_string());
    ///
    /// assert!(map.contains_alias(&alias));
    /// assert!(!map.contains_alias(&CollectionAlias::new("other").unwrap()));
    /// ```
    pub fn contains_alias(
        &self,
        alias: &CollectionAlias,
    ) -> bool {
        self.inner.contains_key(alias)
    }

    /// Clears all aliases from this map.
    ///
    /// # Example
    ///
    /// ```
    /// use kideta_core::collection::{CollectionAlias, CollectionAliasMap};
    ///
    /// let mut map = CollectionAliasMap::new();
    /// map.set(CollectionAlias::new("a").unwrap(), "col_a".to_string());
    /// map.clear();
    /// assert!(map.is_empty());
    /// ```
    pub fn clear(&mut self) {
        self.inner.clear();
    }
}

impl std::fmt::Display for CollectionAliasMap {
    fn fmt(
        &self,
        f: &mut std::fmt::Formatter<'_>,
    ) -> std::fmt::Result {
        write!(f, "{{")?;
        let mut first = true;
        for (alias, target) in &self.inner {
            if !first {
                write!(f, ", ")?;
            }
            write!(f, "{} -> {}", alias, target)?;
            first = false;
        }
        write!(f, "}}")
    }
}

#[cfg(test)]
mod alias_tests {
    use super::*;

    #[test]
    fn test_collection_alias_creation() {
        let alias = CollectionAlias::new("my_collection").unwrap();
        assert_eq!(alias.as_str(), "my_collection");
    }

    #[test]
    fn test_collection_alias_empty_rejected() {
        assert!(CollectionAlias::new("").is_err());
    }

    #[test]
    fn test_alias_map_set_and_get() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("app").unwrap();
        map.set(alias.clone(), "production".to_string());

        assert_eq!(map.get(&alias), Some(&"production".to_string()));
    }

    #[test]
    fn test_alias_map_replace() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("app").unwrap();
        map.set(alias.clone(), "v1".to_string());
        let prev = map.set(alias.clone(), "v2".to_string());

        assert_eq!(prev, Some("v1".to_string()));
        assert_eq!(map.get(&alias), Some(&"v2".to_string()));
    }

    #[test]
    fn test_alias_map_remove() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("temp").unwrap();
        map.set(alias.clone(), "temp_col".to_string());

        let removed = map.remove(&alias);
        assert_eq!(removed, Some("temp_col".to_string()));
        assert!(map.get(&alias).is_none());
    }

    #[test]
    fn test_alias_map_swap() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("products").unwrap();
        map.set(alias.clone(), "products_v1".to_string());

        let old = map.swap(&alias, "products_v2".to_string());

        assert_eq!(old, Some("products_v1".to_string()));
        assert_eq!(map.get(&alias), Some(&"products_v2".to_string()));
    }

    #[test]
    fn test_alias_map_swap_nonexistent() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("new_alias").unwrap();

        let old = map.swap(&alias, "target".to_string());

        assert_eq!(old, None);
        assert_eq!(map.get(&alias), Some(&"target".to_string()));
    }

    #[test]
    fn test_alias_map_iter() {
        let mut map = CollectionAliasMap::new();
        map.set(CollectionAlias::new("a").unwrap(), "col_a".to_string());
        map.set(CollectionAlias::new("b").unwrap(), "col_b".to_string());

        let pairs: Vec<_> = map.iter().collect();
        assert_eq!(pairs.len(), 2);
    }

    #[test]
    fn test_alias_map_contains_alias() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("exists").unwrap();
        map.set(alias.clone(), "col".to_string());

        assert!(map.contains_alias(&alias));
        assert!(!map.contains_alias(&CollectionAlias::new("not_exists").unwrap()));
    }

    #[test]
    fn test_alias_map_clear() {
        let mut map = CollectionAliasMap::new();
        map.set(CollectionAlias::new("a").unwrap(), "col_a".to_string());
        map.clear();
        assert!(map.is_empty());
    }

    #[test]
    fn test_alias_map_display() {
        let mut map = CollectionAliasMap::new();
        map.set(CollectionAlias::new("a").unwrap(), "col_a".to_string());
        map.set(CollectionAlias::new("b").unwrap(), "col_b".to_string());

        let s = format!("{}", map);
        assert!(s.contains("a -> col_a") || s.contains("b -> col_b"));
    }

    #[test]
    fn test_blue_green_deployment_scenario() {
        let mut map = CollectionAliasMap::new();
        let alias = CollectionAlias::new("search").unwrap();

        // Phase 1: Initial deployment with v1
        map.set(alias.clone(), "search_v1".to_string());
        assert_eq!(map.get(&alias), Some(&"search_v1".to_string()));

        // Phase 2: Build v2 in background (alias still points to v1)
        let v2_collection = "search_v2".to_string();

        // Phase 3: Atomic switch - zero downtime
        let old = map.swap(&alias, v2_collection.clone());

        assert_eq!(old, Some("search_v1".to_string()));
        assert_eq!(map.get(&alias), Some(&v2_collection));

        // Phase 4: Safe to delete old collection
        let deleted = map.remove(&alias);
        assert_eq!(deleted, Some("search_v2".to_string()));
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::metric::DistanceMetric;
    use crate::schema::{ScalarField, ScalarFieldType, VectorField};

    #[test]
    fn test_collection_config_creation() {
        let config = CollectionConfig::new("test_collection")
            .add_vector_field(VectorField::new("embedding", 768, DistanceMetric::Cosine))
            .add_scalar_field(ScalarField::new("text", ScalarFieldType::Str));

        assert_eq!(config.name, "test_collection");
        assert_eq!(config.vectors.len(), 1);
        assert_eq!(config.scalar_fields.len(), 1);
    }

    #[test]
    fn test_collection_config_getters() {
        let config = CollectionConfig::new("test").add_vector_field(VectorField::new(
            "vec1",
            128,
            DistanceMetric::L2,
        ));

        assert!(config.get_vector_field("vec1").is_some());
        assert!(config.get_vector_field("nonexistent").is_none());
    }

    #[test]
    fn test_collection_schema_validation() {
        let schema = CollectionSchema::from_config(
            CollectionConfig::new("test").add_vector_field(VectorField::new(
                "embedding",
                768,
                DistanceMetric::Cosine,
            )),
        );
        assert!(schema.validate().is_ok());
    }

    #[test]
    fn test_collection_schema_empty_vectors() {
        let schema = CollectionSchema::from_config(CollectionConfig::new("test"));
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_collection_schema_duplicate_vector_names() {
        let schema = CollectionSchema::from_config(
            CollectionConfig::new("test")
                .add_vector_field(VectorField::new("embedding", 768, DistanceMetric::Cosine))
                .add_vector_field(VectorField::new("embedding", 384, DistanceMetric::L2)),
        );
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_collection_schema_zero_dimension() {
        let schema = CollectionSchema::from_config(
            CollectionConfig::new("test").add_vector_field(VectorField::new(
                "embedding",
                0,
                DistanceMetric::Cosine,
            )),
        );
        assert!(schema.validate().is_err());
    }

    #[test]
    fn test_default_vector_field() {
        let config = CollectionConfig::new("test").add_vector_field(VectorField::new(
            "embedding",
            128,
            DistanceMetric::Cosine,
        ));
        assert_eq!(config.default_vector_field(), Some("embedding"));

        let empty = CollectionConfig::new("empty");
        assert_eq!(empty.default_vector_field(), None);
    }
}
