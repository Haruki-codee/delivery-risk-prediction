-- Run after loading the raw CSV files into the olist schema.
-- 1) Order-level lateness baseline (delivered orders only).
SELECT COUNT(*) AS delivered_orders,
       SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
       ROUND(100.0 * AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;

-- 2) Monthly late-delivery rate.
SELECT DATE_TRUNC('month', order_purchase_timestamp) AS purchase_month,
       COUNT(*) AS orders,
       ROUND(100.0 * AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- 3) Customer state performance.
SELECT c.customer_state, COUNT(*) AS delivered_orders,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders o JOIN olist.customers c USING(customer_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY c.customer_state ORDER BY late_rate_pct DESC;

-- 4) Seller-level delivery performance.
SELECT i.seller_id, COUNT(DISTINCT o.order_id) AS orders,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders o JOIN olist.order_items i USING(order_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY i.seller_id HAVING COUNT(DISTINCT o.order_id) >= 20
ORDER BY late_rate_pct DESC;
