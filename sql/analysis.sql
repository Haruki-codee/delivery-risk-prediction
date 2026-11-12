-- Run after loading the raw CSV files into the olist schema.

-- 1) Order-level lateness baseline (delivered orders only).
SELECT COUNT(*) AS delivered_orders,
       SUM(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1 ELSE 0 END) AS late_orders,
       ROUND(100.0 * AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL;

-- 2) Monthly late-delivery rate and volume.
SELECT DATE_TRUNC('month', order_purchase_timestamp) AS purchase_month,
       COUNT(*) AS orders,
       ROUND(100.0 * AVG(CASE WHEN order_delivered_customer_date > order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders
WHERE order_delivered_customer_date IS NOT NULL
  AND order_estimated_delivery_date IS NOT NULL
GROUP BY 1 ORDER BY 1;

-- 3) Customer state performance ranking.
SELECT c.customer_state,
       COUNT(*) AS delivered_orders,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders o
JOIN olist.customers c USING(customer_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY c.customer_state
ORDER BY late_rate_pct DESC;

-- 4) Interstate (Cross-State) vs Intrastate (Same State) route performance.
SELECT CASE WHEN c.customer_state = s.seller_state THEN 'Intrastate (Same State)'
            ELSE 'Interstate (Cross-Border)' END AS route_type,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct,
       ROUND(AVG(EXTRACT(EPOCH FROM (o.order_delivered_customer_date - o.order_estimated_delivery_date))/86400.0)::numeric, 2) AS avg_delay_days
FROM olist.orders o
JOIN olist.customers c USING(customer_id)
JOIN olist.order_items i USING(order_id)
JOIN olist.sellers s USING(seller_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1;

-- 5) Downstream Customer Review Impact (Satisfaction Drop).
SELECT CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 'Delivered Late'
            ELSE 'Delivered On-Time' END AS delivery_status,
       COUNT(*) AS orders_reviewed,
       ROUND(AVG(r.review_score), 2) AS avg_review_score,
       ROUND(100.0 * AVG(CASE WHEN r.review_score <= 2 THEN 1.0 ELSE 0.0 END), 2) AS pct_negative_reviews,
       ROUND(100.0 * AVG(CASE WHEN r.review_score = 5 THEN 1.0 ELSE 0.0 END), 2) AS pct_five_star_reviews
FROM olist.orders o
JOIN olist.order_reviews r USING(order_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY 1;

-- 6) Seller-level delivery performance (Min 20 orders).
SELECT i.seller_id,
       COUNT(DISTINCT o.order_id) AS orders,
       ROUND(100.0 * AVG(CASE WHEN o.order_delivered_customer_date > o.order_estimated_delivery_date THEN 1.0 ELSE 0.0 END),2) AS late_rate_pct
FROM olist.orders o
JOIN olist.order_items i USING(order_id)
WHERE o.order_delivered_customer_date IS NOT NULL
  AND o.order_estimated_delivery_date IS NOT NULL
GROUP BY i.seller_id
HAVING COUNT(DISTINCT o.order_id) >= 20
ORDER BY late_rate_pct DESC;
